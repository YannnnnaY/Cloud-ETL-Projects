# Data-Engineering-Projects


<img src="https://github.com/YannnnnaY/Data-Engineering-Projects/assets/120424783/7891216d-492a-4cf1-8df1-4a8dd989dc82" width="600" >

#### Apache Airflow

<img src="https://github.com/YannnnnaY/Data-Engineering-Projects/assets/120424783/7b9f5419-e180-4397-80a2-194e73504ae2" width="500" >

#### AWS Setting

EC2 

**Create Instance**

AMI: Ubuntu -> Create new key pair 

-> SELECT "Allow HTTPS traffic from the internet" and "Allow HTTP traffic from the internet"

-> Launch Instance

**Connect to Instance**

Go to the instance -> click "connect" -> "SSH client" -> copy the command and run in terminal
* make sure I'm the only user has access to the key file. Everyone else should have no access. “chmod 600 xxxxx_key.pem"

**Run the ubuntuz_commands.sh**

After get the login user name and password, go to AWS console -> your instance, copy the Public IPv4 DNS -> go to a browser and connect to the address with :8080 

* If not able to connect to the DNS address in the brower, go to the EC2 instance -> Security -> Security groups -> edit inbound rules -> add a rule 
<img width="1000" alt="image" src="https://github.com/YannnnnaY/Data-Engineering-Projects/assets/120424783/5e9b1fe2-3e31-4d7a-8447-3ad2147cdb19">

* AWS Trouble Shooting Doc: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/TroubleshootingInstancesConnecting.html


**Deploying a DAG (Directed Acyclic Graph) script to Apache Airflow - from localhost**

1. DAG Folder Location: ls ~/airflow
2. View the Configuration to find the dags_folder: airflow config get-value core dags_folder
3. Change the dags_folder to the acutal folder name 
4. Copy dag related scripts to the assigned dags_folder
   if deploy airflow from local machine: <img width="400" alt="image" src="https://github.com/YannnnnaY/Data-Engineering-Projects/assets/120424783/c2b4e7bb-4b07-4082-84c8-6217cbd1bea8">
6. kick off the dag on airflow console and check the results
   <img width="1000" alt="image" src="https://github.com/YannnnnaY/Data-Engineering-Projects/assets/120424783/9a4e9dc0-5970-4f2a-90ae-55360c3a0cee">

