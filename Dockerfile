FROM ubuntu

# Set work directory
RUN mkdir -p ./src
WORKDIR ./src
COPY . .

# Install basic dependencies
RUN apt update
RUN apt install nano -y
RUN apt install git -y
RUN apt install curl -y

# Install python3
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN pip3 install --upgrade pip

# Install python dependencies
RUN pip3 install -r requirements.txt