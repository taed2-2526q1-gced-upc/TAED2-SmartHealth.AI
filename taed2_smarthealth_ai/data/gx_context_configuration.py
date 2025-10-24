import great_expectations as gx

from taed2_smarthealth_ai.data.config import INTERIM_DATA_DIR, RAW_DATA_DIR, ROOT_DIR

DATASOURCE_NAME = "pandas_obesity"
RAW_DATA_ASSET = "raw_obesity_data"
INTERIM_DATA_ASSET = "clean_obesity_data"
RAW_DATA_VALIDATOR = "raw_data_validator"
CLEAN_DATA_VALIDATOR = "clean_data_validator"
EXPECTATIONS_SUITE = "obesity_data_suite"
CHECKPOINT = "obesity_checkpoint"

if __name__ == "__main__":
    # 1. create Great Expectations context
    context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)

    # 2. create data docs site
    data_docs_config = {
        "class_name": "SiteBuilder",
        "store_backend": {
            "class_name": "TupleFilesystemStoreBackend",
            "base_directory": "data_docs",
        },
        "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
    }
    context.update_data_docs_site("local_site", data_docs_config)

    # 3. create datasource
    datasource = context.data_sources.add_or_update_pandas(name=DATASOURCE_NAME)

    # 4. create raw and clean assets
    raw_asset = datasource.add_csv_asset(
        name=RAW_DATA_ASSET, filepath_or_buffer=RAW_DATA_DIR / "obesity.csv"
    )
    clean_asset = datasource.add_csv_asset(
        name=INTERIM_DATA_ASSET, filepath_or_buffer=INTERIM_DATA_DIR / "obesity_clean.csv"
    )

    raw_data_batch_definition = raw_asset.add_batch_definition(name="raw_batch")
    clean_data_batch_definition = clean_asset.add_batch_definition(name="clean_batch")

    # 5. create expectations suite
    expectation_suite = gx.ExpectationSuite(EXPECTATIONS_SUITE)
    context.suites.add_or_update(expectation_suite)

    # 6. Add your validation rules (you can add more later)

    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="Gender", value_set=[0, 1])
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="Age", min_value=5, max_value=120)
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Height", min_value=1.0, max_value=2.5
        )
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="Weight", min_value=30, max_value=300)
    )

    # Binary columns
    binary_cols = [
        "family_history_with_overweight",
        "FAVC",
        "SMOKE",
        "SCC",
        "MTRANS_automobile",
        "MTRANS_bike",
        "MTRANS_motorbike",
        "MTRANS_walking",
    ]
    for col in binary_cols:
        expectation_suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=col, value_set=[0, 1])
        )

    # Ordinal scales
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="FCVC", min_value=1, max_value=3)
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="NCP", min_value=1, max_value=4)
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="CAEC", min_value=0, max_value=3)
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="CH2O", min_value=1, max_value=3)
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="FAF", min_value=0, max_value=3)
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="TUE", min_value=0, max_value=2)
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="CALC", min_value=0, max_value=3)
    )

    # Target variable (depends on dataset)
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Obesity", value_set=[0, 1, 2, 3, 4, 5, 6]
        )
    )

    expectation_suite.save()

    # 7. create validator definitions
    raw_data_validation_definition = gx.ValidationDefinition(
        name=RAW_DATA_VALIDATOR, data=raw_data_batch_definition, suite=expectation_suite
    )
    clean_data_validation_definition = gx.ValidationDefinition(
        name=CLEAN_DATA_VALIDATOR, data=clean_data_batch_definition, suite=expectation_suite
    )

    context.validation_definitions.add_or_update(raw_data_validation_definition)
    context.validation_definitions.add_or_update(clean_data_validation_definition)

    # 8. create checkpoint
    action_list = [gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")]
    validation_definitions = [clean_data_validation_definition]

    checkpoint = gx.Checkpoint(
        name=CHECKPOINT,
        validation_definitions=validation_definitions,
        actions=action_list,
        result_format="SUMMARY",
    )

    context.checkpoints.add_or_update(checkpoint)

    print("Great Expectations configuration completed successfully!")
