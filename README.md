# LLAMATUI-WEB-SERVER

ONLY CONSOLE I(NO X) INTERFACE TO LLAMA.CPP SERVER TO SERVER WEB API  

HOW TO RUN THE PROGRAM

FIRST WITH PYTHON3 INSTALLED AND NO COMPILATION:

deinflating .zip

to home/"user"/entorno  (THIS ZIP MUST BE IN /home/"user"/entorno , wont run if you rename or put this in other directory)

activate /entorno/bin/activate

VERY IMPORTANT:!!!  PSUTIL INSTALL 

firSt install this library: psutil

pip install psutil 

recommended but not neccesary***************

export llama server to your system path for can run the llamatui  from every directory if you want ,otherwise ,   you can select the directory of llama-server from the program 


for example:

export PATH="$PATH:/home/namesys/llama.cpp-master/build/bin"

**************************************************

and now you can run python3 serverllama.py


**********HOW TO COMPILE THE BINARY

compile it if you want binary for your system
pip install pyinstaller

pyinstaller --onefile llamatui.py

the file would be written do /dist 
assign execution permission with:
chmod +x ./llamatui

you can rename for name do you want.

execute , and navigate with 


ALSO I INCLUDE IN THE DIRECTORY debian 12amd64 binary the program compiled yet 
for this architecture amd64 debian and ubuntu only need permision and run



LLAMATUI , YOU CAN MOVE WITH TAB AND ENTER TO CHANGE SETTINGS , YOU CAN DEBUG THE SERVER WITH V KEY , START S , KILL , K , SELECG GGUF , CHANGE IP AND PORT..ETC
YOU CAN CHANGE THE LAYERS LOADED TO CPU  , THREADS DEPENDING ON YOUR CPU CORES ,ETC , AND YOU CAN SEE THE VRAM CONSUMPTION WITH N KEY.





