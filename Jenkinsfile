pipeline {
    agent any

    environment {
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        CONDA_PATH = "${env.HOME}/miniconda3/etc/profile.d/conda.sh"
        CONDA_ENV = "mlops-agp"
    }

    options {
        timestamps()
        ansiColor('xterm')
        timeout(time: 60, unit: 'MINUTES')
    }

    stages {

        stage('Preprocess') {
            steps {
                echo "Running preprocessing..."
                sh """
                source ${CONDA_PATH}
                conda activate ${CONDA_ENV}
                python preprocess.py
                """
            }
        }

        stage('Train & Evaluate') {
            steps {
                echo "Starting training..."
                script {
                    env.TRAIN_START = sh(script: "date +%s", returnStdout: true).trim()
                    sh """
                    source ${CONDA_PATH}
                    conda activate ${CONDA_ENV}
                    python trainandevaluate.py 2>&1 | tee train_log.txt
                    """
                    env.TRAIN_END = sh(script: "date +%s", returnStdout: true).trim()
                }
            }
        }

        stage('Record retrain time metric') {
            steps {
                echo "Recording training duration..."
                sh """
                source ${CONDA_PATH}
                conda activate ${CONDA_ENV}
                python - <<'PY'
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
                echo "Deploying model..."
                sh """
                source ${CONDA_PATH}
                conda activate ${CONDA_ENV}
                python deploy.py
                """
            }
        }

        stage('Run Flask App') {
            steps {
                echo "Starting Flask app..."
                sh """
                source ${CONDA_PATH}
                conda activate ${CONDA_ENV}
                nohup python app.py > flask_app.log 2>&1 &
                """
            }
        }

    }

    post {
        success {
            echo "Pipeline completed successfully!"
        }
        failure {
            echo "Pipeline failed. Check logs for details."
        }
        always {
            echo "Cleaning up workspace..."
            // Optionally, stop Flask app if needed
            sh "pkill -f 'python app.py' || true"
        }
    }
}
