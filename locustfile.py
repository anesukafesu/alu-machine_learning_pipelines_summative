from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    
    @task(1)
    def train(self):
        self.client.get("/train")
    
    @task(2)
    def predict(self):
        self.client.get("/predict?loan_int_rate=15&person_home_ownership=RENT&loan_percent_income=20&cb_person_default_on_file=N")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)