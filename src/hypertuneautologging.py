from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.datasets import load_breast_cancer
import mlflow
import dagshub

dagshub.init(repo_owner='thehashiramasenju', repo_name='mlopsexperimentspractice', mlflow=True)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name = 'target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators' : [10, 20, 30, 50, 90, 100],
    'max_depth' : [None, 10, 20, 30, 50, 100, 110]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs = 1, verbose=2)
 
#mlflow configurations
mlflow.set_tracking_uri("https://dagshub.com/TheHashiramaSenju/MLOPsExperimentsPractice.mlflow")
mlflow.set_experiment("breast-cancer-RandomForest-hp")

mlflow.autolog()
with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)
    #this is responsible for logging all the params and metrics for each of the iterations in grid search cv =, the top is parent we start as child nd make things happen here 
    
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])
    
    #loading the best params
    #Displaying the best params through grid search
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_
    best_index = grid_search.best_index_
    
    
       
    #logging test and train datas
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
   
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    
    mlflow.set_tag("Author", "Darshan")
    
    print(best_params)
    print()
    print(best_score)