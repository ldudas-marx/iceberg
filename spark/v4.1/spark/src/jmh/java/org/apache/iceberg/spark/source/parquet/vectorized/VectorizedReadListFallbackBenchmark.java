/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.spark.source.parquet.vectorized;

import static org.apache.iceberg.types.Types.NestedField.optional;
import static org.apache.spark.sql.functions.array;
import static org.apache.spark.sql.functions.col;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableProperties;
import org.apache.iceberg.hadoop.HadoopTables;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.relocated.com.google.common.collect.Maps;
import org.apache.iceberg.spark.source.IcebergSourceBenchmark;
import org.apache.iceberg.types.Types;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Threads;

/**
 * Benchmark comparing vectorized and row-by-row Parquet reads of a wide schema that includes a list
 * column.
 *
 * <p>To run this benchmark for spark-4.1: <code>
 *   ./gradlew -DsparkVersions=4.1 :iceberg-spark:iceberg-spark-4.1_2.13:jmh \
 *       -PjmhIncludeRegex=VectorizedReadListFallbackBenchmark \
 *       -PjmhOutputPath=benchmark/results.txt
 * </code>
 */
public class VectorizedReadListFallbackBenchmark extends IcebergSourceBenchmark {

  private static final int NUM_INT_COLUMNS = 999;
  private static final String LIST_COLUMN = "list_col";
  private static final int BATCH_SIZE = 5000;
  private static final int NUM_FILES = 1;
  private static final int NUM_ROWS_PER_FILE = 10_000;

  private static final String[] INT_COLUMNS = intColumnNames();

  @Setup
  public void setupBenchmark() {
    setupSpark();
    appendData();
  }

  @TearDown
  public void tearDownBenchmark() throws IOException {
    tearDownSpark();
    cleanupFiles();
  }

  @Override
  protected Configuration initHadoopConf() {
    return new Configuration();
  }

  @Override
  protected Table initTable() {
    List<Types.NestedField> fields = Lists.newArrayListWithCapacity(NUM_INT_COLUMNS + 1);
    for (int i = 0; i < NUM_INT_COLUMNS; i++) {
      fields.add(optional(i + 1, INT_COLUMNS[i], Types.IntegerType.get()));
    }
    // One unsupported-for-vectorization column: a list of ints.
    fields.add(
        optional(
            NUM_INT_COLUMNS + 1,
            LIST_COLUMN,
            Types.ListType.ofOptional(NUM_INT_COLUMNS + 2, Types.IntegerType.get())));
    Schema schema = new Schema(fields);

    PartitionSpec partitionSpec = PartitionSpec.unpartitioned();
    HadoopTables tables = new HadoopTables(hadoopConf());
    return tables.create(schema, partitionSpec, parquetWriteProps(), newTableLocation());
  }

  private Map<String, String> parquetWriteProps() {
    Map<String, String> properties = Maps.newHashMap();
    properties.put(TableProperties.METADATA_COMPRESSION, "gzip");
    return properties;
  }

  private void appendData() {
    for (int fileNum = 0; fileNum < NUM_FILES; fileNum++) {
      Column[] columns = new Column[NUM_INT_COLUMNS + 1];
      for (int i = 0; i < NUM_INT_COLUMNS; i++) {
        columns[i] = col("id").plus(i).cast("int").as(INT_COLUMNS[i]);
      }
      // Small list so the 999 int columns dominate the read cost being measured.
      columns[NUM_INT_COLUMNS] =
          array(col("id").cast("int"), col("id").plus(1).cast("int")).as(LIST_COLUMN);

      Dataset<Row> df = spark().range(NUM_ROWS_PER_FILE).select(columns);
      appendAsFile(df);
    }
  }

  @Benchmark
  @Threads(1)
  public void readAllColumnsNonVectorized() {
    withTableProperties(
        vectorizationProps(false),
        () -> {
          Dataset<Row> df = spark().read().format("iceberg").load(table().location());
          materialize(df);
        });
  }

  @Benchmark
  @Threads(1)
  public void readAllColumnsVectorized() {
    withTableProperties(
        vectorizationProps(true),
        () -> {
          // The list column is vectorized along with the 999 int columns.
          Dataset<Row> df = spark().read().format("iceberg").load(table().location());
          materialize(df);
        });
  }

  private static Map<String, String> vectorizationProps(boolean enabled) {
    Map<String, String> properties = Maps.newHashMap();
    properties.put(TableProperties.PARQUET_VECTORIZATION_ENABLED, String.valueOf(enabled));
    properties.put(TableProperties.PARQUET_BATCH_SIZE, String.valueOf(BATCH_SIZE));
    return properties;
  }

  private static String[] intColumnNames() {
    String[] names = new String[NUM_INT_COLUMNS];
    for (int i = 0; i < NUM_INT_COLUMNS; i++) {
      names[i] = "int_" + i;
    }
    return names;
  }
}
