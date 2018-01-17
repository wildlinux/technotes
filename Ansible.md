
<h1 > Ansible </h1>

<!-- TOC -->

- [1. Overview](#1-overview)
- [2. Ad-hoc commands](#2-ad-hoc-commands)
- [3. Inventory](#3-inventory)
- [4. Playbook](#4-playbook)
- [5. Variants](#5-variants)
- [6. Replay part of playbook](#6-replay-part-of-playbook)

<!-- /TOC -->

# 1. Overview

Ansbile use ssh to excute shell command, or run ansbile modules at remote/target machine, which could be used to automation the installation and deployment process.

Please ref to <http://docs.ansible.com/ansible/latest/intro_installation.html> for installation guide.
 
# 2. Ad-hoc commands

Get some instinct of ansbile by run any command like the following, where 10.9.208.41 could be any IP or hostname, and vagrant is a user of the given host.

```bash
$ansible 10.9.208.41 -u vagrant --ask-pass -a "/sbin/ifconfig" 
```

# 3. Inventory

Edit /etc/ansible/hosts as the following

```ymal

[dbserver1] 
192.168.21.135 ansible_user=root 

[dbserver2]
192.168.21.134 ansible_user=root

[dbservers:children]
dbserver1
dbserver2

```

Try again the following command:

```
$ansible dbservers --ask-pass -a "/sbin/ifconfig"
```

# 4. Playbook

A simple playbook may look like the following:

The beginning part tell to which host, should this playbook to be applied: dbservers, which will be parsed from /etc/ansible/hosts

The 'copy' part is one task, which use ansible module 'copy', which has parameters like 'src/dest/owner/mode' etc.
"{{ localinstalldir }} are pre-defined parameters, with_fileglob + src:{{ item }}, will let ansible to loop over each item meet with_fileglob.

The last 'yum' part is to install packages defined in "{{ ext_arts }}"

```yaml


- hosts: [dbservers]
  vars_files:
    - ./global_vars.yml

- name: Get the addtional packages    
  copy:
    src: "{{ item }}"   # a space after {{ may cause error
    dest: "{{ localinstalldir }}/Base-os/"
    owner: "root"
    mode: 0744
  with_fileglob: "{{ Installer_Source_dir }}/Base-os/*"
  tags: getfiles  

- debug: var=playme 

- name : Install the addtional packages
  yum: 
    name: "{{ localinstalldir }}/Base-os/{{ item }}"
    mode: 0777
    owener: root
    state: present
  with_items:  "{{ ext_arts }}"
  tags: preinstall 
```

```sh
ansible-playbook playbook.yml -f 10 --list-hosts --ask-become-pass
```

# 5. Variants

Variants could be defined like the this:

```yaml

localinstalldir: "/RPMInstall"                             
Installer_Source_dir: "/somelocaldir/files-Installers"      ext_arts:  
  - compat-libstdc++-33-3.2.3-72.el7.i686.rpm
  - compat-libstdc++-33-3.2.3-72.el7.x86_64.rpm 

```    

# 6. Replay part of playbook

How to excute only part of the playbook with tags

command和tags必须对齐。

```
- command: echo task111  
  tags: task1  
- command: echo task112  
  tags: task1  
- command: echo task333  
  tags:  
     - task3 
     - task1 
- command: echo task222  
  tags:  
     - task2  
```

```
ansible-playbook test2.yml --skip-tags="play1"  

ansible-playbook test2.yml --tags="task2,task3"
```

`http://blog.csdn.net/tcxp_for_wife/article/details/41685695`
