#!/bin/sh
export M2_HOME=/usr1/apache-maven-3.3.3
export GRADLE_HOME=/usr1/gradle-3.3
export JAVA_HOME=/usr1/jdk1.8.0_111
export CI_ROOT=/usr1/CloudDragon/tools/CodeDEX/plugins/CodeDEX
export FORTIFY_HOME=$CI_ROOT/tool/fortify
export COVERITY_HOME=$CI_ROOT/tool/coverity
export CODEDEX_TOOL=$CI_ROOT/tool
export PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$GRADLE_HOME/bin:$FORTIFY_HOME/bin:$COVERITY_HOME/bin:$PATH
export FORTIFY_BUILD_ID=a
 
export inter_dir=$WORKSPACE/inter_dir 
export cov_tmp_dir=$inter_dir/cov_tmp
export for_tmp_dir=$inter_dir/for_tmp
export project_root=$WORKSPACE/
 
rm -rf $inter_dir
cd $project_root
#生成coverity中间件
cov-build --dir "$cov_tmp_dir" mvn clean install
#通过jar转换成fortify中间件
java -jar $CODEDEX_TOOL/transferfortify-1.3.1.jar "java" "$FORTIFY_BUILD_ID" "$inter_dir" 
 
cd $cov_tmp_dir
$CODEDEX_TOOL/7za a -tzip coverity.zip * -r
mv coverity.zip "$inter_dir"
cd $for_tmp_dir
$CODEDEX_TOOL/7za a -tzip fortify.zip * -r
mv fortify.zip "$inter_dir"
