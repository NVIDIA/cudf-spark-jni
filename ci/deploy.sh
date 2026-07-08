#!/bin/bash
#
# Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

###
# Script to deploy spark-rapids-jni jar files along with other classifiers,
# such as cudaXXX, sources, javadoc.
#
# Argument(s):
#   SIGN_FILE: true/false, whether to sign the jar/pom file to de deployed
#
# Used environment(s):
#   OUT_PATH:       The path where jar files are
#   SIGN_TOOL:      Tools to sign files, e.g., gpg, nvsec, only required when $1 is 'true'
#   CLASSIFIERS:    The classifier list of the jars to be deployed
#   SERVER_ID:      The repository id for this deployment.
#   SERVER_URL:     The URL where to deploy artifacts.
#   GPG_PASSPHRASE: The passphrase used to sign files, only required when <SIGN_TOOL> is gpg
#   POM_FILE:       Project pom file to be deployed
#   MVN_SETTINGS:   Maven configuration file
#
###

set -ex

SIGN_FILE=$1
SIGN_TOOL=${SIGN_TOOL:-"gpg"}
OUT_PATH=${OUT_PATH:-"target"}
POM_FILE=${POM_FILE:-"pom.xml"}
MVN_SETTINGS=${MVN_SETTINGS:-"ci/settings.xml"}

MVN="mvn -Dmaven.wagon.http.retryHandler.count=3 -DretryFailedDeploymentCount=3 -B -s $MVN_SETTINGS"
REL_ARTIFACTID=$($MVN exec:exec -f $POM_FILE -q --non-recursive -Dexec.executable=echo -Dexec.args='${project.artifactId}')
REL_VERSION=$($MVN exec:exec -f $POM_FILE -q --non-recursive -Dexec.executable=echo -Dexec.args='${project.version}')

echo "REL_VERSION: $REL_VERSION, OUT_PATH: $OUT_PATH \
        SERVER_URL: $SERVER_URL, SERVER_ID: $SERVER_ID"

###### Build types/files from classifiers ######
FPATH="$OUT_PATH/$REL_ARTIFACTID-$REL_VERSION"
CLASS_TYPES=''
CLASS_FILES=''
ORI_IFS="$IFS"
IFS=','
for CLASS in $CLASSIFIERS; do
    CLASS_TYPES="${CLASS_TYPES},jar"
    CLASS_FILES="${CLASS_FILES},${FPATH}-${CLASS}.jar"
done
# Remove the first char ','
CLASS_TYPES=${CLASS_TYPES#*,}
CLASS_FILES=${CLASS_FILES#*,}
IFS="$ORI_IFS"

###### Copy jar so we strip off classifier  #######
# Use the first classifier(aka jar file) as the default jar
FIRST_FILE=${CLASS_FILES%%,*}
cp -f "$FIRST_FILE" "$FPATH.jar"

GPG_PLUGIN="org.apache.maven.plugins:maven-gpg-plugin:3.1.0:sign-and-deploy-file"
###### Build the deploy command ######
if [ "$SIGN_FILE" == true ]; then
    case $SIGN_TOOL in
        nvsec)
            DEPLOY_CMD="$MVN $GPG_PLUGIN -Dgpg.executable=nvsec_sign"
            ;;
        gpg)
            DEPLOY_CMD="$MVN $GPG_PLUGIN -Dgpg.passphrase=$GPG_PASSPHRASE "
            ;;
        *)
            echo "Error unsupported sign type : $SIGN_TYPE !"
            echo "Please set variable SIGN_TOOL 'nvsec'or 'gpg'"
            exit 255
            ;;
    esac
else
    DEPLOY_CMD="$MVN -B deploy:deploy-file"
fi

DEPLOY_CMD="$DEPLOY_CMD -Durl=$SERVER_URL -DrepositoryId=$SERVER_ID -DpomFile=$POM_FILE"
echo "Deploy CMD: $DEPLOY_CMD"

###### Deploy spark-rapids-jni jar with all its additions ######
$DEPLOY_CMD -Dfile=$FPATH.jar \
            -DpomFile=$POM_FILE \
            -Dsources=$FPATH-sources.jar \
            -Djavadoc=$FPATH-javadoc.jar \
            -Dfiles=$CLASS_FILES \
            -Dtypes=$CLASS_TYPES \
            -Dclassifiers=$CLASSIFIERS
