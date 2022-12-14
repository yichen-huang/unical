
SRC	=	.cu

COMMON_SRC =	$(addprefix src/, $(SRC))

NAME	=	heat_transfer

CC	=	nvcc

CPPFLAGS	=	-Wall -Wextra

all:	$(NAME)

$(NAME):
		$(CC) -o $(NAME) $(COMMON_SRC)

test:
		nvprof ./$(NAME)

clean:
		$(RM) $(OBJ)

fclean:	clean
		$(RM) $(NAME)

re:	fclean all

.PHONY:	all clean fclean re test