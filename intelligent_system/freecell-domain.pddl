(define (domain freecell)
  	(:requirements :strips :typing)  
  	(:types card num suit)
  	(:predicates
		; CARDS CHARACTERISTICS
	    (VALUE ?c -card ?v -num)
	    (SUCCESSOR ?n1 -num ?n0 -num)
	    (SUIT ?c -card ?s -suit)
	    (CANSTACK ?c1 -card ?c2 -card)
		; INIT CARDS & FINAL OBJECTIVE CARDS
	    (HOME ?c -card)
		; CELL CHARACTERISTICS
	    (CELLSPACE ?n -num) ; # of cells
	    (COLSPACE ?n -num) ; cols max height
		; CARDS STATUS
  		(ON ?c1 -card ?c2 -card) 
	    (CLEAR ?c -card) ; can stack
	    (BOTTOMCOL ?c -card)
		; CARD STATUS INTERNAL USE
	    (INCELL ?c -card)
	)

	; NOTATION CONVENTION
	; NOTES: EXAMPLE OF ?moving_card ?card_sender ?card_receiver
	; 	C4		CA
	; 	 |
	; 	C2
	; 		||
	; 		\/
	; 	C4		CA
	; 			|
	; 			C2
	; HERE
	; CA: ?card_receiver
	; C2: ?moving_card
	; C4: ?card_sender

	; NOTE:
	; 	CELLSPACE AND COLSPACE DECREASE WHEN WE ADD A CARD TO 
	; 	AND INCREASE WHEN WE REMOVE CARD

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;	STACKED FUNCTIONS
;
;		STACKED TO NEW COL, FREECELL, HOME
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	(:action move_stacked_card_to_another_stack
        :parameters (
			?moving_card ?card_sender ?card_receiver -card
			?moving_card_num ?card_receiver_num -num
		)
        :precondition (and
			; CHECK IF CARD CAN BE MOVED
			(CLEAR ?moving_card)
			(ON ?moving_card ?card_sender)
			; CHECK IF CARD CAN BE RECEIVED
			(CLEAR ?card_receiver)
			(SUCCESSOR ?card_receiver_num ?moving_card_num)
            (CANSTACK ?moving_card ?card_receiver)
		)
        :effect (and
			; UPDATE CARD_SENDER
		    (not (ON ?moving_card ?card_sender))
		    (CLEAR ?card_sender)
			; UPDATE CARD_RECEIVER
		    (not (CLEAR ?card_receiver))
			; UPDATE MOVING_CARD
		    (ON ?moving_card ?card_receiver)
        )
    )

	(:action move_stacked_card_to_freecell
		:parameters (
			?moving_card ?card_sender -card
			?max_cell_available ?cell_to_occupy -num
		)
		:precondition (and
			; CHECK IF MOVING_CARD CAN BE MOVED
			(CLEAR ?moving_card)
			(ON ?moving_card ?card_sender)
			; CHECK IF max_cell_available IS INDEED THE MAXIMUM NUMBER OF CELLS AVAILABLE
			(CELLSPACE ?max_cell_available)
			; CHECK IF cell_to_occupy IS max_cell_available-1 (PDDL ALWAYS TAKE MAX-1 don't know why)
			(SUCCESSOR ?max_cell_available ?cell_to_occupy)
		)
		:effect (and
			; UPDATE CARD_SENDER
			(not (ON ?moving_card ?card_sender))
			(CLEAR ?card_sender)
			; UPDATE MONVING_CARD
			(not (CLEAR ?moving_card))
			(INCELL ?moving_card)
			; UPDATE CELLS
			(not (CELLSPACE ?max_cell_available))
			(CELLSPACE ?cell_to_occupy)
		)
	)
	
	(:action move_stacked_card_to_home
		:parameters (
			?moving_card ?card_sender ?home_card -card
			?moving_card_num ?home_card_num -num
			?suit -suit
		)
		:precondition (and
			; CHECK IF CARD CAN BE MOVED
			(CLEAR ?moving_card) 
			(ON ?moving_card ?card_sender)
			; CHECK IF moving_card AND home_card ARE FROM TEH SAME SUIT
			(SUIT ?moving_card ?suit)
			(SUIT ?home_card ?suit)
			; CHECK IF HOME CONDITIONS ARE MET
			(HOME ?home_card)
			(VALUE ?moving_card ?moving_card_num)
			(VALUE ?home_card ?home_card_num)
			(SUCCESSOR ?moving_card_num ?home_card_num)
		)
	 	:effect (and
			; UPDATE CARD_SENDER
		    (not (ON ?moving_card ?card_sender))
		    (CLEAR ?card_sender)
			; UPDATE MOVING_CARD
		    (not (CLEAR ?moving_card))
			(HOME ?moving_card)
			; UPDATE HOME_CARD
            (not (HOME ?home_card))
		)
	)

	(:action move_stacked_card_to_new_col
		:parameters (
			?moving_card ?card_sender -card
			?max_col_available ?col_to_occupy -num
		)
		:precondition (and
			; CHECK IF MOVING_CARD CAN BE MOVED
			(CLEAR ?moving_card)
			(ON ?moving_card ?card_sender)
			; CHECK IF max_col_available IS AVAILABLE
			(COLSPACE ?max_col_available)
			; CHECK IF col_to_occupy IS max_col_available+1 (increases, don't know why)
			(SUCCESSOR ?max_col_available ?col_to_occupy)
		)
		:effect (and
			; UPDATE CARD_SENDER
			(not (ON ?moving_card ?card_sender))
			(CLEAR ?card_sender)
			; UPDATE MONVING_CARD
			(BOTTOMCOL ?moving_card)
			; UPDATE COLS
			(not (COLSPACE ?max_col_available))
			(COLSPACE ?col_to_occupy)
		)
	)
	
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;	INDIVIDUAL FUNCTIONS
;
;		INDIVIDUAL TO HOME, FREECELL, STACK, NEW COL
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	(:action move_individual_card_to_home
		:parameters (
			?moving_card ?home_card -card
			?moving_card_num ?home_card_num -num
			?max_col_available ?col_to_occupy -num
			?suit -suit
		)
		:precondition (and
			; CHECK IF MOVING_CARD CAN BE MOVED
			(CLEAR ?moving_card)
			(BOTTOMCOL ?moving_card)
			; CHECK IF MOVING_CARD AND HOME_CARD ARE FROM TEH SAME SUIT
			(SUIT ?moving_card ?suit)
			(SUIT ?home_card ?suit)
			; CHECK IF HOME_CARD CONDITIONS ARE MET
			(HOME ?home_card)
			(VALUE ?moving_card ?moving_card_num)
			(VALUE ?home_card ?home_card_num)
			(SUCCESSOR ?moving_card_num ?home_card_num)
			; CHECK IF max_col_available IS AVAILABLE
			(COLSPACE ?max_col_available)
			; CHECK IF col_to_occupy IS max_col_available+1 (increases, don't know why)
			(SUCCESSOR ?col_to_occupy ?max_col_available)
		)
	 	:effect (and
			; UPDATE MOVING_CARD
		    (not (CLEAR ?moving_card))
			(not (BOTTOMCOL ?moving_card))
			(HOME ?moving_card)
			; UPDATE HOME_CARD
            (not (HOME ?home_card))
			; UPDATE COLS
			(not (COLSPACE ?max_col_available))
			(COLSPACE ?col_to_occupy)
		)
	)

	(:action move_individual_card_to_stack
		:parameters (
			?moving_card ?card_receiver -card
			?moving_card_num ?card_receiver_num -num
			?max_col_available ?col_to_occupy -num
		)
		:precondition (and
			; CHECK IF MOVING_CARD CAN BE MOVED
			(CLEAR ?moving_card)
			(BOTTOMCOL ?moving_card)
			; CHECK IF MOVING_CARD CAN BE RECEIVED
			(CLEAR ?card_receiver)
			(SUCCESSOR ?card_receiver_num ?moving_card_num)
            (CANSTACK ?moving_card ?card_receiver)
			; CHECK IF max_col_available IS AVAILABLE
			(COLSPACE ?max_col_available)
			; CHECK IF col_to_occupy IS max_col_available+1 (increases, don't know why)
			(SUCCESSOR ?col_to_occupy ?max_col_available)
		)
		:effect (and
			; UPDATE CARD_RECEIVER
            (not (CLEAR ?card_receiver))
			; UPDATE MOVING_CARD
			(not (BOTTOMCOL ?moving_card))
			(ON ?moving_card ?card_receiver)
			; UPDATE COLS
			(not (COLSPACE ?max_col_available))
			(COLSPACE ?col_to_occupy)
		)
	)

	(:action move_individual_card_to_freecell
		:parameters (
			?moving_card -card
			?max_cell_available ?cell_to_occupy -num
			?max_col_available ?col_to_occupy -num
		)
		:precondition (and
			; CHECK IF CARD CAN BE MOVED
			(CLEAR ?moving_card)
			(BOTTOMCOL ?moving_card)
			; CHECK IF max_cell_available IS INDEED THE MAXIMUM NUMBER OF CELLS AVAILABLE
			(CELLSPACE ?max_cell_available)
			; CHECK IF cell_to_occupy IS max_cell_available-1 (PDDL ALWAYS TAKE MAX-1 don't know why)
			(SUCCESSOR ?max_cell_available ?cell_to_occupy)
			; CHECK IF max_col_available IS AVAILABLE
			(COLSPACE ?max_col_available)
			; CHECK IF col_to_occupy IS max_col_available+1 (increases, don't know why)
			(SUCCESSOR ?col_to_occupy ?max_col_available)
		)
		:effect (and
			; UPDATE MOVING_CARD
			(not (BOTTOMCOL ?moving_card))
			(not (CLEAR ?moving_card))
			(INCELL ?moving_card)
			; UPDATE CELLS
			(not (CELLSPACE ?max_cell_available))
			(CELLSPACE ?cell_to_occupy)
			; UPDATE COLS
			(not (COLSPACE ?max_col_available))
			(COLSPACE ?col_to_occupy)
		)
	)
	
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;	FREECELL FUNCTIONS
;
;		FREECELL TO HOME OR ANOTHER STACK, OR NEW COL
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	(:action move_card_from_freecell_to_home
	 	:parameters (
			?moving_card ?home_card -card
			?moving_card_num ?home_card_num -num
			?max_cell_available ?current_cell -num
			?suit -suit
		)
	 	:precondition (and
			; CHECK IF MOVING_CARD CAN BE MOVED AND IS IN FREECELL
			(INCELL ?moving_card)
			(CELLSPACE ?max_cell_available)
			(SUCCESSOR ?current_cell ?max_cell_available)
			; CHECK IF MOVING_CARD AND HOME_CARD ARE FROM TEH SAME SUIT
			(SUIT ?moving_card ?suit)
			(SUIT ?home_card ?suit)
			; CHECK IF HOME CONDITIONS ARE MET
			(HOME ?home_card)
			(VALUE ?moving_card ?moving_card_num)
			(VALUE ?home_card ?home_card_num)
			(SUCCESSOR ?moving_card_num ?home_card_num)
		)
	 	:effect (and
			; UPDATE FREECELL
			(not (CELLSPACE ?max_cell_available))
			(CELLSPACE ?current_cell)
			; UPDATE MOVING_CARD
			(not (INCELL ?moving_card))
			(HOME ?moving_card)
			; UPDATE HOME_CARD
			(not (HOME ?home_card))
		)
	)

	(:action move_card_from_freecell_to_stack
		:parameters (
			?moving_card ?card_receiver -card
			?moving_card_num ?card_receiver_num -num
			?max_cell_available ?current_cell -num
		)
		:precondition (and
			; CHECK IF MOVING_CARD CAN BE MOVED AND IS IN FREECELL
			(INCELL ?moving_card)
			(CELLSPACE ?max_cell_available)
			(SUCCESSOR ?current_cell ?max_cell_available)
			; CHECK IF CARD CAN BE RECEIVED
			(CLEAR ?card_receiver)
			(SUCCESSOR ?card_receiver_num ?moving_card_num)
            (CANSTACK ?moving_card ?card_receiver)
		)
		:effect (and
			; UPDATE FREECELL
			(not (CELLSPACE ?max_cell_available))
			(CELLSPACE ?current_cell)
			; UPDATE CARD_RECEIVER
		    (not (CLEAR ?card_receiver))
			; UPDATE MOVING_CARD
			(not (INCELL ?moving_card))
		    (ON ?moving_card ?card_receiver)
			(CLEAR ?moving_card)
		)
	)

	; TIME OUT ON 10-4
	; (:action move_card_from_freecell_to_new_col
	; 	:parameters (
	; 		?moving_card -card
	; 		?max_cell_available ?current_cell -num
	; 		?max_col_available ?col_to_occupy -num
	; 	)
	; 	:precondition (and
	; 		; CHECK IF MOVING_CARD CAN BE MOVED AND IS IN FREECELL
	; 		(INCELL ?moving_card)
	; 		(CELLSPACE ?max_cell_available)
	; 		(SUCCESSOR ?current_cell ?max_cell_available)
	; 		; CHECK IF max_col_available IS AVAILABLE
	; 		(COLSPACE ?max_col_available)
	; 		; CHECK IF col_to_occupy IS max_col_available+1 (increases, don't know why)
	; 		(SUCCESSOR ?max_col_available ?col_to_occupy)
	; 	)
	; 	:effect (and
	; 		; UPDATE FREECELL
	; 		(not (CELLSPACE ?max_cell_available))
	; 		(CELLSPACE ?current_cell)
	; 		; UPDATE COLS
	; 		(not (COLSPACE ?max_col_available))
	; 		(COLSPACE ?col_to_occupy)
	; 		; UPDATE MONVING_CARD
	; 		(not (INCELL ?moving_card))
	; 		(BOTTOMCOL ?moving_card)
	; 		(CLEAR ?moving_card)
	; 	)
	; )
)