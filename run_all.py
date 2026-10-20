"""One command to run the whole pipeline:  python run_all.py"""
from src import generate_data, eda, train, statistical_analysis, predict, config

if not config.DATA_PATH.exists():
    config.DATA_PATH.parent.mkdir(exist_ok=True)
    generate_data.make_data().to_csv(config.DATA_PATH, index=False)
print("== EDA =="); eda.run()
print("\n== TRAINING =="); train.main()
print("\n== STATISTICS =="); statistical_analysis.run()
print("\n== PREDICTION DEMO ==")
print(predict.predict([{"area": 1800, "bedrooms": 3, "bathrooms": 2, "age": 5, "distance_city": 8, "city": "Pune"}]))
