
SRC1	=	standard_monolithic.cu
SRC2	=	standard_grid_stride.cu
SRC3	=	unified_monolithic.cu
SRC4	=	unified_grid_stride.cu

COMMON_SRC1 =	$(addprefix src/, $(SRC1))
COMMON_SRC2 =	$(addprefix src/, $(SRC2))
COMMON_SRC3 =	$(addprefix src/, $(SRC3))
COMMON_SRC4 =	$(addprefix src/, $(SRC4))

NAME1	=	standard_monolithic
NAME2	=	standard_grid_stride
NAME3	=	unified_monolithic
NAME4	=	unified_grid_stride

CC	=	nvcc

CPPFLAGS	=	-Wall -Wextra

all:	$(NAME)

$(NAME):
		$(CC) -o $(NAME1) $(COMMON_SRC1)
		$(CC) -o $(NAME2) $(COMMON_SRC2)
		$(CC) -o $(NAME3) $(COMMON_SRC3)
		$(CC) -o $(NAME4) $(COMMON_SRC4)

test:
		nvprof ./$(NAME1)
		nvprof ./$(NAME2)
		nvprof ./$(NAME3)
		nvprof ./$(NAME4)

clean:
		$(RM) $(OBJ)

fclean:	clean
		$(RM) $(NAME1)
		$(RM) $(NAME2)
		$(RM) $(NAME3)
		$(RM) $(NAME4)

re:	fclean all

.PHONY:	all clean fclean re test