


## sock5 based kcp and ssh

```
mac$ more start_kcptun.sh 
./client_darwin_amd64 -r "kcpserverIP:port" -l ":20022" -mode fast2

mac$ more start_s5.sh 
#!/usr/bin/expect -f
spawn ssh -D 1080 -N -f yourname@127.0.0.1 -p 20022
expect "password:"
send "breath@BRIDGE\r"
interact

Then you can setup your brower with sock5 proxy of 127.0.0.1:1080

```

## chisel

It works at the beginning, but is blocked later.

https://github.com/jpillora/chisel#performance

On the server:

```sh

 196  apt-get update
  197  apt-get -y upgrade
  198  wget https://dl.google.com/go/go1.10.3.linux-amd64.tar.gz
  199  tar -xvf go1.10.3.linux-amd64.tar.gz 
  200  df -h
  201  mv go /usr/local/
  202  export GOROOT=/usr/local/go
  203  export GOPATH=$HOME/Projects/Proj1
  204  export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
  205  vi ./.profile                      to add the above three export
  206  go version
  207  go env
  208  go get -v github.com/jpillora/chisel
  209  ls
  210  cd Projects/
  211  ls
  212  cd Proj1/
  213  ls
  214  cd src/
  215  ls
  216  cd ../bin/
  217  ls
  218  cd chise
  219  ls -l
  223  chisel server -p 8080 --socks5 --key supersecret
       it will print the finger print of the server

  249  iptables -F
  250  iptables -X
  251  iptables -L
  252  ./iptables-init.sh 
  253  iptables -L
  254  iptables -P INPUT DROP
  255  iptables -L
  more iptables-init.sh
iptables  -t nat -A POSTROUTING -s 10.8.0.0/24 -o ens3 -j MASQUERADE
iptables -A INPUT -i lo -p all -j ACCEPT
iptables -A INPUT -p all -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A INPUT -i tun+ -j ACCEPT
iptables -A FORWARD -i tun+ -j ACCEPT
iptables -A INPUT -p tcp -m tcp --dport 8080 -j ACCEPT
iptables -A INPUT -p tcp -m tcp --dport 48023 -j ACCEPT
iptables -A INPUT -p udp --dport 8999 -j ACCEPT
iptables -A INPUT -s 10.8.0.0/24 -p all -j ACCEPT
iptables -A FORWARD -d 10.8.0.0/24 -j ACCEPT
iptables -A INPUT -p tcp -m tcp --dport 22 -j ACCEPT
iptables -P INPUT DROP

```

On the client

```

533  sudo brew install go
  534  brew install go
  535  go get -v github.com/jpillora/chisel
  536  ls
  537  cd ..
  538  ls
  539  cd ..
  540  ls
  541  cd go/
  542  ls
  543  cd bin
  544  ls
  545  ./chisel client --fingerprint THE_FINGER_PRINT_OF_YOUR_SERVER SERVER_IP:8080 socks

  Then setup the sock5 proxy of your Browser.

```