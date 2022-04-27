# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from datetime import datetime

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator
from airflow.providers.databricks.operators.databricks_repos import (
    DatabricksReposUpdateOperator,
)

default_args = {
    'owner': 'airflow',
    'databricks_conn_id': 'databricks',
}

with DAG(
    dag_id='anomaly_detection_dag',
    start_date=datetime(2022, 4, 27),
    schedule_interval='@yearly',
    default_args=default_args,
    tags=['databricks', 'anomaly_detection'],
    catchup=False,
) as dag:

    # Example of updating a Databricks Repo to the latest code
    repo_path = "/Repos/sean.wilkinson@databricks.com/anomaly_detection"
    update_repo = DatabricksReposUpdateOperator(task_id='update_repo', repo_path=repo_path, branch="main")

    notebook_task_params = {
        'new_cluster': {
            'spark_version': '10.4.x-cpu-ml-scala2.12',
            'node_type_id': 'n1-standard-4',
            'num_workers': 1,
        },
        'notebook_task': {
            'notebook_path': f'{repo_path}/notebooks/hello_airflow',
            'base_parameters': {'output': 'Hello world!'},
        },
    }

    # Call a one time notebook task in Databricks. 
    notebook_task = DatabricksSubmitRunOperator(task_id='notebook_task', json=notebook_task_params)

    (update_repo >> notebook_task)