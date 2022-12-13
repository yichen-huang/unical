SRC	=	heatransfer_allegro4.cpp

OBJ	=	$(SRC:.cpp=.o)

NAME	=	heat_transfer

CC	=	g++

CPPFLAGS	=	-Wall -Wextra

ALLEGRO_FLAG	=	-I/usr/include/allegro4 -L/usr/lib -lalleg

all:	$(NAME)

$(NAME):	$(OBJ)
		$(CC) $(OBJ) -o $(NAME) $(ALLEGRO_FLAG) -std=c++11

clean:
		$(RM) $(OBJ)

fclean:	clean
		$(RM) $(NAME)

re:	fclean all

.PHONY:	all clean fclean re