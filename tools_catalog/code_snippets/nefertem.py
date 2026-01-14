import nefertem


def data_validation(data_file_path):
    # Set configurations
    output_path = "./nefertem_run"
    store = {"name": "local", "store_type": "local"}
    data_resource = {
        "name": "resource_name",
        "path": data_file_path,
        "store": "local",
    }
    run_config = {
        "operation": "validation",
        "exec_config": [{"framework": "frictionless"}],
    }

    # Create a client and run
    client = nefertem.create_client(output_path=output_path, store=[store])
    with client.create_run([data_resource], run_config) as run:
        run.validate()
        run.log_report()
        run.persist_report()
