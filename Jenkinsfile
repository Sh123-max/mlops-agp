pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        CONDA_ENV = "${env.CONDA_ENV ?: 'mlops-agp'}"  // Name of your pre-created conda env
    }

    options {
        timestamps()
        ansiColor('xterm')
        timeout(time: 1, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {
        stage('Checkout SCM') {
            steps {
                git branch: 'main', url: 'https://github.com/your-username/mlops-agp.git'
            }
        }

        stage('Activate Conda Environment') {
            steps {
                echo "Activating pre-configured conda environment: ${CONDA_ENV}"
                sh """
                    source ~/miniconda3/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}
                    python -c "import sys, numpy, pandas, sklearn, xgboost, mlflow; print('Python:', sys.version); print('numpy', numpy.__version__)"
                """
            }
        }

        stage('Preprocess Data') {
            steps {
                echo "Running preprocess.py"
                sh """
                    source ~/miniconda3/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}
                    python preprocess.py 2>&1 | tee preprocess_log.txt
                """
            }
        }

        stage('Train Model') {
            steps {
                script {
                    env.TRAIN_START = sh(script: """
                        source ~/miniconda3/etc/profile.d/conda.sh
                        conda activate ${CONDA_ENV}
                        python - <<'PY'
import time
print(int(time.time()))
PY
                    """, returnStdout: true).trim()

                    sh """
                        source ~/miniconda3/etc/profile.d/conda.sh
                        conda activate ${CONDA_ENV}
                        python trainandevaluate.py 2>&1 | tee train_log.txt
                    """

                    env.TRAIN_END = sh(script: """
                        source ~/miniconda3/etc/profile.d/conda.sh
                        conda activate ${CONDA_ENV}
                        python - <<'PY'
import time
print(int(time.time()))
PY
                    """, returnStdout: true).trim()
                }
            }
        }

        stage('Record Retrain Time Metric') {
            steps {
                script {
                    sh """
                        source ~/miniconda3/etc/profile.d/conda.sh
                        conda activate ${CONDA_ENV}
                        python - <<'PY'
import os
start = int(os.environ.get('TRAIN_START', '0'))
end = int(os.environ.get('TRAIN_END', '0'))
t = end - start if (start and end) else 0
with open('retrain_time.txt', 'w') as f:
    f.write(str(t))
print("retrain_time_seconds:", t)
PY
                    """
                }
            }
        }

        stage('Deploy Application') {
            steps {
                echo "Running deploy.py & app.py"
                sh """
                    source ~/miniconda3/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}
                    python deploy.py 2>&1 | tee deploy_log.txt
                    # Optionally start Flask app in background (for demo/testing)
                    nohup python app.py &
                """
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: "train_log.txt,retrain_time.txt,preprocess_log.txt,deploy_log.txt,${MODEL_DIR}/**", fingerprint: true
            }
        }
    }

    post {
        always {
            echo "Pipeline finished. Cleaning workspace..."
            cleanWs()
        }
        success {
            echo "Pipeline succeeded."
        }
        failure {
            echo "Pipeline failed."
        }
    }
}
