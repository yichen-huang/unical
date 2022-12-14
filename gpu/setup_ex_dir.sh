#!/bin/sh

MAKEFILE_CONTENT='
SRC	=	.cu\n
\n
COMMON_SRC =	$(addprefix src/, $(SRC))\n
\n
NAME	=\n
\n
CC	=	nvcc\n
\n
CPPFLAGS	=	-Wall -Wextra\n
\n
all:	$(NAME)\n
\n
$(NAME):\n
		$(CC) -o $(NAME) $(COMMON_SRC)\n
\n
test:\n
		nvprof ./$(NAME)\n
\n
clean:\n
		$(RM) $(OBJ)\n
\n
fclean:	clean\n
		$(RM) $(NAME)\n
\n
re:	fclean all\n
\n
.PHONY:	all clean fclean re test
'

WORKSTATION_ADDRESS='USER=\
IP_ADDRESS=\n
PORT=\n
\n
WORKSTATION_PATH=\n
LOCAL_MOUNT_POINT=\n'
SCRIPT_HEADER='#!/bin/sh\n
. ./script/address.cfg\n'
CONNECT_WORKSTATION='ssh $USER@$IP_ADDRESS -p $PORT'
MOUNT_WORKSTATION='sshfs $USER@$IP_ADDRESS:$WORKSTATION_PATH $LOCAL_MOUNT_POINT -p $PORT'
UNMOUNT_WORKSTATION='fusermount -u $LOCAL_MOUNT_POINT'

create_dir() {
    local name=$1

    mkdir $name;
    mkdir $name/script;
    mkdir $name/${name}_mount_point;
    mkdir $name/src;
    mkdir $name/example;
}

create_makefile() {
    local path=$1

    touch $path/Makefile;
    echo $MAKEFILE_CONTENT > $path/Makefile;
}

for exercise_name in "$@"
do
    create_dir $exercise_name;
    create_makefile $exercise_name ;
done