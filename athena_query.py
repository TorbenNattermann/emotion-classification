import boto3
import pandas as pd


class AthenaQuery:

    def __init__(self):
        self.client = boto3.client('athena')
        self.Database = 'dm_pantheon_context_classification'
        self.output = 's3://pdm-pantheon-preprocessed/athena_queries_tmp/'

    def athena_request(self):
        query = """
            SELECT ref_referrer_id, ref_adslot_id, ref_variant_id, label
            FROM dm_pantheon_context_classification.row_ceres_contextclf20231204
            LIMIT 2000000;
           """
        response = self.client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={
                'Database': self.Database
            },
            ResultConfiguration={
                'OutputLocation': self.output,
            }
        )

        query_execution_id = response['QueryExecutionId']

        # Wait for the query to finish execution
        while True:
            query_status = self.client.get_query_execution(
                QueryExecutionId=query_execution_id
            )['QueryExecution']['Status']['State']

            if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break

        if query_status == 'SUCCEEDED':
            # Query succeeded, fetch the results
            query_results = self.client.get_query_results(
                QueryExecutionId=query_execution_id
            )

            # Convert the results to DataFrame
            column_names = [col['Name'] for col in query_results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            data_rows = []
            for row in query_results['ResultSet']['Rows'][1:]:
                row_values = [data['VarCharValue'] for data in row['Data']]
                data_rows.append(row_values)

            df = pd.DataFrame(data_rows, columns=column_names)

            # Save DataFrame as Parquet file locally
            df.to_parquet('query_results.parquet', index=False)
        else:
            print("Query execution failed or was cancelled.")

    # Example usage
if __name__ == "__main__":
    athena_query = AthenaQuery()
    athena_query.athena_request()
