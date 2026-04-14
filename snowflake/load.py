import snowflake.connector
import os


conn = snowflake.connector.connect(
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    warehouse="AIRBYTE_WAREHOUSE",
    database="AIRBYTE_DATABASE",
    schema="PUBLIC",
)

# Load detections from S3 into FactDetection
conn.cursor().execute(f"""
    COPY INTO FactDetection
    FROM s3://{os.environ["S3_BUCKET"]}/detections/
    CREDENTIALS=(
        AWS_KEY_ID='{os.environ["AWS_ACCESS_KEY_ID"]}'
        AWS_SECRET_KEY='{os.environ["AWS_SECRET_ACCESS_KEY"]}'
    )
    FILE_FORMAT=(TYPE=CSV);
""")

conn.close()
