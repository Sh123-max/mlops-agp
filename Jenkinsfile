pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        CONDA_PATH = "/var/lib/jenkins/miniconda3"
        CONDA_ENV = "mlops-agp"
    }

    options {
        timeout(time: 1, unit: 'HOURS') // maximum pipeline runtime
        timestamps()
        ansiColor('xterm')
    }

    stages {

        stage('Preprocess') {
            steps {
                echo 'Running preprocessing...'
                sh """
                #!/bin/bash
                source ${CONDA_PATH}/etc/profile.d/conda.sh
                conda activate ${CONDA_ENV}
                python3 preprocess.py
                """
            }
        }

        stage('Train & Evaluate') {
            steps {
                echo 'Running training and evaluation...'
                script {
                    env.TRAIN_START = sh(script: """#!/bin/bash
                    python3 - <<'PY'
import time
print(int(time.time()))
PY
                    """, returnStdout: true).trim()

                    sh """
                    #!/bin/bash
                    source ${CONDA_PATH}/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}
                    python3 trainandevaluate.py 2>&1 | tee train_log.txt
                    """

                    env.TRAIN_END = sh(script: """#!/bin/bash
                    python3 - <<'PY'
import time
print(int(time.time()))
PY
                    """, returnStdout: true).trim()
                }
            }
        }

        stage('Record retrain time metric') {
            steps {
                echo 'Recording retrain time...'
                sh """
                #!/bin/bash
                python3 - <<'PY'
import os
start = int(os.environ.get('TRAIN_START', '0'))
end = int(os.environ.get('TRAIN_END', '0'))
t = end - start if (start and end) else 0
with open('retrain_time.txt','w') as f:
    f.write(str(t))
print("retrain_time_seconds:", t)
PY
                """
                archiveArtifacts artifacts: 'train_log.txt,retrain_time.txt', fingerprint: true
            }
        }

        stage('Deploy Model') {
            steps {
                echo 'Deploying model...'
                sh """
                #!/bin/bash
                source ${CONDA_PATH}/etc/profile.d/conda.sh
                conda activate ${CONDA_ENV}
                python3 deploy.py
                """
            }
        }

        stage('Run Flask App') {
            steps {
                echo 'Starting Flask app...'
                sh """
                #!/bin/bash
                source ${CONDA_PATH}/etc/profile.d/conda.sh
                conda activate ${CONDA_ENV}
                pkill -f app.py || true
                nohup python3 app.py &
                """
            }
        }
    }

    post {
        always {
            echo 'Cleaning up workspace...'
            sh 'pkill -f python app.py || true'
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for details.'
        }
    }
}
