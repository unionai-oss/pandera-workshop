import union

import pandas as pd
import pandera as pa
import typing
from sklearn.ensemble import RandomForestClassifier
from pandera.typing import DataFrame


image = union.ImageSpec.from_env(
    name="pandera-workshop",
    packages=[
        "flytekitplugins-pandera",
        "scikit-learn",
        "pyarrow",
        "pandera",
        "numpy",
    ],
)


class RawData(pa.DataFrameModel):
    age: int = pa.Field(in_range={"min_value": 0, "max_value": 200})
    sex: int = pa.Field(isin=[0, 1])
    cp: int = pa.Field(
        isin=[
            1,  # typical angina
            2,  # atypical angina
            3,  # non-anginal pain
            4,  # asymptomatic
        ]
    )
    trestbps: int = pa.Field(in_range={"min_value": 0, "max_value": 200})
    chol: int = pa.Field(in_range={"min_value": 0, "max_value": 600})
    fbs: int = pa.Field(isin=[0, 1])
    restecg: int = pa.Field(
        isin=[
            0,  # normal
            1,  # having ST-T wave abnormality
            2,  # showing probable or definite left ventricular hypertrophy by Estes' criteria
        ]
    )
    thalach: int = pa.Field(in_range={"min_value": 0, "max_value": 300})
    exang: int = pa.Field(isin=[0, 1])
    oldpeak: float = pa.Field(in_range={"min_value": 0, "max_value": 10})
    slope: int = pa.Field(
        isin=[
            1,  # upsloping
            2,  # flat
            3,  # downsloping
        ]
    )
    ca: int = pa.Field(isin=[0, 1, 2, 3])
    thal: int = pa.Field(
        isin=[
            3,  # normal
            6,  # fixed defect
            7,  # reversible defect
        ]
    )
    target: int = pa.Field(ge=0, le=4)

    class Config:
        coerce = True


class ParsedData(RawData):
    target: int = pa.Field(isin=[0, 1])


class TrainingData(ParsedData):
    @pa.dataframe_check(error="Patients with heart disease should have higher average cholesterol")
    def validate_cholesterol(cls, df: pd.DataFrame) -> bool:
        healthy_chol = df[df.target == 0].chol.mean()
        disease_chol = df[df.target == 1].chol.mean()
        return disease_chol > healthy_chol

    @pa.dataframe_check(error="Patients with heart disease should not have lower max heart rate (thalach) on average")
    def validate_max_heart_rate(cls, df: pd.DataFrame) -> bool:
        healthy_thalach = df[df.target == 0].thalach.mean()
        disease_thalach = df[df.target == 1].thalach.mean()
        return disease_thalach < healthy_thalach

    @pa.dataframe_check(error="Exercise-induced angina is not more common in disease group")
    def validate_exercise_induced_angina(cls, df: pd.DataFrame) -> bool:
        exang_ratio = df[df.target == 1].exang.mean() / df[df.target == 0].exang.mean()
        return exang_ratio > 2.0

    @pa.dataframe_check(error="cp, exang, and oldpeak should be positively correlated with target")
    def validate_feature_correlations(cls, df: pd.DataFrame) -> bool:
        """Ensure key feature correlations with target remain strong"""
        corrs = df.corr()['target']
        return all(corrs[['cp', 'exang', 'oldpeak']] > 0.2)  # These should be strongly correlated


DataSplits = typing.NamedTuple(
    "DataSplits",
    training_set=DataFrame[ParsedData],
    test_set=DataFrame[ParsedData],
)

@union.task(container_image=image, enable_deck=True)
def parse_raw_data(raw_data: DataFrame[RawData]) -> DataFrame[ParsedData]:
    """Convert the target to a binary target."""
    print("parsing raw data")
    return raw_data.assign(target=lambda _: (_.target > 0).astype(int))

@union.task(container_image=image, enable_deck=True)
def split_data(
    parsed_data: DataFrame[ParsedData],
    test_size: float,
    random_state: int,
) -> DataSplits:
    print("splitting data")
    training_set = parsed_data.sample(frac=test_size, random_state=random_state)
    test_set = parsed_data[~parsed_data.index.isin(training_set.index)]
    return training_set, test_set

def get_features_and_target(dataset):
    X = dataset[[x for x in dataset if x != "target"]]
    y = dataset["target"]
    return X, y

@union.task(container_image=image, enable_deck=True)
def train_model(training_set: DataFrame[TrainingData], random_state: int) -> RandomForestClassifier:
    print("training model")
    print(training_set.corr()["target"][['cp', 'exang', 'oldpeak']])
    print(all(training_set.corr()["target"][['cp', 'exang', 'oldpeak']] > 0.2))
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    X, y = get_features_and_target(training_set)
    model.fit(X, y)
    return model

@union.task(container_image=image)
def evaluate_model(model: RandomForestClassifier, test_set: pd.DataFrame) -> float:
    test_features, test_target = get_features_and_target(test_set)
    acc = (model.predict(test_features) == test_target).mean()
    return float(acc)

@union.workflow
def training_pipeline(
    dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 100
) -> tuple[RandomForestClassifier, float]:
    parsed_data = parse_raw_data(dataset)
    training_set, test_set = split_data(parsed_data, test_size, random_state)
    model = train_model(training_set, random_state)
    acc = evaluate_model(model, test_set)
    return model, acc
