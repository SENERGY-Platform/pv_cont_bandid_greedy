FROM amancevice/pandas:1.5.1-alpine
LABEL org.opencontainers.image.source https://github.com/SENERGY-Platform/pv-greedy-operator
WORKDIR /usr/src/app
COPY . .
RUN apk update && apk add git && git log -1 --pretty=format:"commit=%H%ndate=%cd%n" > git_commit && python3 -m pip install --no-cache-dir -r requirements.txt && rm -rf .git
CMD [ "python3", "-u", "main.py"]
