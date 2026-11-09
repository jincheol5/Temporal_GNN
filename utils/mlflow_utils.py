import os
import mlflow
from pathlib import Path

BASE_PATH=os.path.join("..","data","Temporal_GNN") # 상대 경로
os.makedirs(BASE_PATH,exist_ok=True)
class MLFlowUtils:
    @staticmethod
    def setup(experiment_name:str):
        ### Tracking DB 경로 설정
        # sqlite:/// -> 상대 경로, sqlite://// -> 절대 경로 
        tracking_path=os.path.join(BASE_PATH,"tracking")
        os.makedirs(tracking_path,exist_ok=True)
        tracking_db_path=f"sqlite:///{tracking_path}/experiments.db" 
        mlflow.set_tracking_uri(tracking_db_path)

        ### Artifact 저장 경로
        # Artifact 저장 경로는 Experiment를 생성할 때 설정.
        # MLflow에서는 Tracking Store와 Artifact Store가 분리되어 있지만,
        # 어느 Artifact Store를 사용할지는 Tracking Store에 저장된 Experiment 정보가 결정함
        # Artifact 저장 경로는 Experiment의 메타데이터로서 Tracking Store에 기록됨
        # Tracking Store에는 Artifact 저장 경로 정보가 저장되고, Artifact Store에는 실제 모델 가중치 파일이 저장됨
        artifact_path=os.path.join(BASE_PATH,"artifact")
        os.makedirs(artifact_path,exist_ok=True)

        ### Experiment 조회 및 없을 시 생성
        # mlflow.set_tracking_uri()로 지정한 Tracking Store에서 Experiment를 조회
        experiment=mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=Path(artifact_path).resolve().as_uri() # Artifact 저장 경로 설정
            )

        ### Experiment 설정
        mlflow.set_experiment(experiment_name=experiment_name)
