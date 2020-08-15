#### Installation


##### System Dependencies

- redis
- Opencv


```shell
# ubuntu 18.04
sudo apt-get install python3-pip libopencv-dev redis-server -y
pip3 install opencv-python
```

##### Python Dependencies

```shell
cd rl
pip3 install -r requirements.txt

# install with pip 
pip3 install -e . 
```

> Note: XingTian only tested a lot in Tensorflow 1.15. Other versions may have unknown problems. Please let us know if there are any problems. 

