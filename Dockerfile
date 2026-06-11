FROM ubuntu:latest
LABEL authors="ekatm"

ENTRYPOINT ["top", "-b"]