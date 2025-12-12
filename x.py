import joblib, pandas as pd

bundle = joblib.load("pipeline.joblib")
pipe = bundle["pipeline"]
features = bundle["features"]

x = {f: None for f in features}
x.update({
    "school":"GP","sex":"F","age":18,"address":"U","famsize":"GT3","Pstatus":"T",
    "Medu":4,"Fedu":4,"Mjob":"teacher","Fjob":"teacher","reason":"course","guardian":"mother",
    "traveltime":2,"studytime":2,"failures":0,"schoolsup":"yes","famsup":"no","paid":"no",
    "activities":"yes","nursery":"yes","higher":"yes","internet":"yes","romantic":"no",
    "famrel":4,"freetime":3,"goout":4,"Dalc":1,"Walc":1,"health":3,"absences":4,"G1":14,"G2":15
})

X = pd.DataFrame([x], columns=features)
print(pipe.predict(X), pipe.predict_proba(X))
