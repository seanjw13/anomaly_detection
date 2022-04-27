# anomaly_detection
This example demonstrates how to use the [Databricks Operators for Airflow](https://airflow.apache.org/docs/apache-airflow-providers-databricks/stable/_api/airflow/providers/databricks/operators/databricks/index.html) to create a simple DAG that parameterizes and calls a Databricks notebook.

**Instructions for running on MacOS**

1. Install Docker Desktop. By default Docker Desktop limits memory usage to 2GB. Airflow will need at least 4GB to run successfully. You can increase this limit [here](https://docs.docker.com/desktop/mac/#resources).
2. Clone this repository.
3. This examples uses the [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/start/docker.html#running-airflow-in-docker) example with minimal modifications. cd to the anomaly_detection/airflow folder and then run the CLI command `docker-compose up airflow-init` .
4. Run the CLI command `docker-compose up` to start up the containers running Airflow.
5. Navigate to http://localhost:8080/ in your web browser. It will ask you to log in. Username: airflow, Password: airflow.
6. Add a [Databricks Connection to Airflow](https://airflow.apache.org/docs/apache-airflow-providers-databricks/stable/connections/databricks.html). Navigate to 'Admin' then 'Connections'. Add a new connection with the conn_id of 'databricks' that contains the Databricks Personal Access Token (PAT) to use.