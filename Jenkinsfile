pipeline {
    agent any

    environment {
        # base defaults - Jenkins will override per-stage for parallel runs
        BASE_MODEL_DIR = "${env.BASE_MODEL_DIR ?: 'models'}"
        BASE_DATA_DIR  = "${env.BASE_DATA_DIR ?: 'data'}"
        PROJECT_NAME = "${env.PROJECT_NAME ?: 'diabetes'}"
        DATA_DIR    = "${env.DATA_DIR ?: 'data'}"
        MODEL_DIR   = "${env.MODEL_DIR ?: 'models'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
        CONDA_PATH  = "/home/shreekar/miniconda3"
        CONDA_ENV   = "mlops-agp"
        # names for systemd services used to run flask apps
        DIABETES_SERVICE = "mlops-diabetes.service"
        HEART_SERVICE    = "mlops-heart.service"
    }

    options {
        timeout(time: 1, unit: 'HOURS')
        timestamps()
        ansiColor('xterm')
    }

    stages {
        stage('Check changes (data)') {
            steps {
                script {
                    sh "git fetch --all --quiet || true"
                    def changedFiles = sh(script: "git diff --name-only HEAD~1 HEAD || true", returnStdout: true).trim()
                    echo "Changed files (HEAD~1..HEAD):\n${changedFiles}"
                }
            }
        }

        stage('Run Both Projects (diabetes & heart)') {
            parallel {
                stage('diabetes') {
                    environment {
                        PROJECT_NAME = "diabetes"
                        DATA_DIR = "${env.BASE_DATA_DIR}/diabetes"
                        MODEL_DIR = "${env.BASE_MODEL_DIR}/diabetes"
                    }
                    steps {
                        echo "Running preprocessing + training for DIABETES (DATA_DIR=${env.DATA_DIR}, MODEL_DIR=${env.MODEL_DIR})..."
                        sh """
                            set -e
                            mkdir -p ${DATA_DIR} ${MODEL_DIR}
                            . ${CONDA_PATH}/etc/profile.d/conda.sh
                            conda activate ${CONDA_ENV}

                            export PROJECT_NAME=${PROJECT_NAME}
                            export DATA_DIR=${DATA_DIR}
                            export MODEL_DIR=${MODEL_DIR}
                            export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

                            python3 preprocess.py
                            python3 trainandevaluate.py 2>&1 | tee diabetes_train_log.txt || true
                        """
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'diabetes_train_log.txt', allowEmptyArchive: true, fingerprint: true
                            sh "cp -v ${MODEL_DIR}/last_run_summary.json . || true"
                            archiveArtifacts artifacts: 'last_run_summary.json', allowEmptyArchive: true
                        }
                    }
                }

                stage('heart') {
                    environment {
                        PROJECT_NAME = "heart"
                        DATA_DIR = "${env.BASE_DATA_DIR}/heart"
                        MODEL_DIR = "${env.BASE_MODEL_DIR}/heart"
                    }
                    steps {
                        echo "Running preprocessing + training for HEART (DATA_DIR=${env.DATA_DIR}, MODEL_DIR=${env.MODEL_DIR})..."
                        sh """
                            set -e
                            mkdir -p ${DATA_DIR} ${MODEL_DIR}
                            . ${CONDA_PATH}/etc/profile.d/conda.sh
                            conda activate ${CONDA_ENV}

                            export PROJECT_NAME=${PROJECT_NAME}
                            export DATA_DIR=${DATA_DIR}
                            export MODEL_DIR=${MODEL_DIR}
                            export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

                            python3 preprocess.py
                            python3 trainandevaluate.py 2>&1 | tee heart_train_log.txt || true
                        """
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'heart_train_log.txt', allowEmptyArchive: true, fingerprint: true
                            sh "cp -v ${MODEL_DIR}/last_run_summary.json . || true"
                            archiveArtifacts artifacts: 'last_run_summary.json', allowEmptyArchive: true
                        }
                    }
                }
            }
        }

        stage('Preprocess (Selected Project)') {
            steps {
                echo "Preprocessing SELECTED PROJECT for deployment: ${env.PROJECT_NAME}"
                sh """
                    set -e
                    . ${CONDA_PATH}/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}
                    export PROJECT_NAME=${PROJECT_NAME}
                    export DATA_DIR=${BASE_DATA_DIR}/${PROJECT_NAME}
                    export MODEL_DIR=${BASE_MODEL_DIR}/${PROJECT_NAME}
                    python3 preprocess.py
                """
            }
        }

        stage('Train & Evaluate (Selected Project)') {
            steps {
                echo "Running training for SELECTED DEPLOYMENT PROJECT: ${env.PROJECT_NAME}"
                script {
                    env.TRAIN_START = sh(script: "python3 -c 'import time; print(int(time.time()))'", returnStdout: true).trim()
                    sh """
                        set -e
                        . ${CONDA_PATH}/etc/profile.d/conda.sh
                        conda activate ${CONDA_ENV}
                        export PROJECT_NAME=${PROJECT_NAME}
                        export DATA_DIR=${BASE_DATA_DIR}/${PROJECT_NAME}
                        export MODEL_DIR=${BASE_MODEL_DIR}/${PROJECT_NAME}
                        export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
                        python3 trainandevaluate.py 2>&1 | tee train_log.txt || true
                    """
                    env.TRAIN_END = sh(script: "python3 -c 'import time; print(int(time.time()))'", returnStdout: true).trim()
                }
            }
        }

        stage('Model Validation') {
            steps {
                echo "Validating model performance against previous best..."
                sh """
                    set -e
                    . ${CONDA_PATH}/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}

                    python3 - <<'PY'
import json, os, sys
MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
summary_path = os.path.join(MODEL_DIR, 'last_run_summary.json')
if not os.path.exists(summary_path):
    print('No summary file found at:', summary_path)
    with open('validation_failure.txt', 'w') as f:
        f.write('No last_run_summary.json found. Check train_log.txt for errors.')
    sys.exit(1)

summary = json.load(open(summary_path))
best = summary.get('best', {})
should_deploy = best.get('should_deploy', True)
deploy_reason = best.get('deploy_reason', 'No reason provided')

if not should_deploy:
    print(f'Model validation blocked: {deploy_reason}')
    with open('validation_failure.txt', 'w') as f:
        f.write(deploy_reason)
    sys.exit(1)

print(f'Model validation passed: {deploy_reason}')
PY
                """
            }
            post {
                failure {
                    echo "Model validation failed"
                    archiveArtifacts artifacts: 'validation_failure.txt,train_log.txt', allowEmptyArchive: true, fingerprint: true
                }
            }
        }

        stage('Ensemble Creation Check') {
            steps {
                echo "Checking ensemble creation..."
                sh """
                    python3 - <<'PY'
import json, os
MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
summary_path = os.path.join(MODEL_DIR, 'last_run_summary.json')
if os.path.exists(summary_path):
    summary = json.load(open(summary_path))
    best = summary.get('best', {})
    if best.get('is_ensemble', False):
        print('ENSEMBLE MODEL CREATED:', best.get('name', ''))
        open('ensemble_created.txt', 'w').write('Ensemble created: ' + best.get('name', ''))
    else:
        print('No ensemble created')
else:
    print('No summary file found')
PY
                """
            }
            post {
                always {
                    archiveArtifacts artifacts: 'ensemble_created.txt', allowEmptyArchive: true, fingerprint: true
                }
            }
        }

        stage('Record Retrain Time Metric') {
            steps {
                sh """
                    python3 - <<'PY'
import os
start = int(os.environ.get('TRAIN_START', '0'))
end = int(os.environ.get('TRAIN_END', '0'))
t = end - start if (start and end) else 0
open('retrain_time.txt','w').write(str(t))
print("Retrain time:", t)
PY
                """
                archiveArtifacts artifacts: 'train_log.txt,retrain_time.txt', allowEmptyArchive: true, fingerprint: true
            }
        }

        stage('Deploy Both Projects and Restart Services') {
            steps {
                echo "Deploying both diabetes & heart models to their respective MODEL_DIRs and restarting services"
                sh """
                    set -e
                    . ${CONDA_PATH}/etc/profile.d/conda.sh
                    conda activate ${CONDA_ENV}

                    # deploy diabetes
                    export PROJECT_NAME=diabetes
                    export MODEL_DIR=${BASE_MODEL_DIR}/diabetes
                    export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
                    python3 deploy.py --project diabetes --stage Staging || true

                    # deploy heart
                    export PROJECT_NAME=heart
                    export MODEL_DIR=${BASE_MODEL_DIR}/heart
                    python3 deploy.py --project heart --stage Staging || true

                    # restart services (requires permission). If you don't have sudo in Jenkins,
                    # either run these commands manually or grant Jenkins sudo rights for these commands.
                    if command -v sudo >/dev/null 2>&1; then
                        sudo systemctl restart ${DIABETES_SERVICE} || true
                        sudo systemctl restart ${HEART_SERVICE} || true
                    else
                        echo "sudo not found - please restart ${DIABETES_SERVICE} and ${HEART_SERVICE} manually"
                    fi
                """
            }
            post {
                always {
                    archiveArtifacts artifacts: 'models/**/model_metadata.json', allowEmptyArchive: true, fingerprint: true
                }
            }
        }

        stage('Run Flask App Health Checks') {
            steps {
                echo "Checking both Flask services health endpoints..."
                sh """
                    set -e
                    # diabetes at 5000
                    if curl --max-time 5 --silent --fail http://localhost:5000/health; then
                        echo "Diabetes app healthy"
                    else
                        echo "Diabetes app not healthy"
                        exit 1
                    fi

                    # heart at 5005
                    if curl --max-time 5 --silent --fail http://localhost:5005/health; then
                        echo "Heart app healthy"
                    else
                        echo "Heart app not healthy"
                        exit 1
                    fi
                """
            }
        }

    }

    post {
        always {
            echo "Cleaning up workspace..."
            cleanWs()
        }
    }
}
