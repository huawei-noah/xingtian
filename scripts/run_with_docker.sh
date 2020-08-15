#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
#GPU=$1
name=${USER}_xt_cpu_${HASH}

echo "Launching container named '${name}' on CPU..."
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi
#cmd=docker

# run with starcraft env 
#${cmd} run -i \
#    --network host \
#    --name $name \
#    -v `pwd`:/pymarl-src \
#    -v /var/run/docker.sock:/var/run/docker.sock \
#    -v /usr/bin/docker:/usr/bin/docker \
#    -v /home/xw/sim/StarCraftII:/pymarl-sim/StarCraftII \
#    -t xt_py36_torch1.2_tf1.8_rl_sim:v1 \
#    ${@:2}

#--network host \
#${cmd} run -i \
#    --name $name \
#    -v `pwd`:/xt-src \
#    -v /var/run/docker.sock:/var/run/docker.sock \
#    -v /usr/bin/docker:/usr/bin/docker \
#    -t xt_py36_torch1.2_tf1.8_rl_sim:v2 \
#    ${@:2}
${cmd} run -i \
    --name $name \
    -v `pwd`:/xt-src \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -t xt_py36_pytorch-1.2_tf-1.15_rl_sim:v2 \
    ${@:2}
