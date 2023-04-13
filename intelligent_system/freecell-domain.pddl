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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;	STACKED FUNCTIONS
;
;		THEY ONLY WORKS WITH STACKED CARDS
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	(:action move_stacked_card_to_freecell
		:parameters (
			?moving_card ?card_sender -card
			?max_cell_available ?cell_to_occupy -num
		)
		:precondition (and
			; CHECK IF CARD CAN BE MOVED
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


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;	INDIVIDUAL FUNCTIONS
;
;		THEY ONLY IN CASE YOU HAVE ONE CARD LEFT
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	(:action move_individual_card_to_home
		:parameters (
			?moving_card ?home_card -card
			?moving_card_num ?home_card_num -num
			?suit -suit
		)
		:precondition (and
			; CHECK IF CARD CAN BE MOVED
			(CLEAR ?moving_card)
			(BOTTOMCOL ?moving_card)
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
			; UPDATE MOVING_CARD
		    (not (CLEAR ?moving_card))
			(not (BOTTOMCOL ?moving_card))
			(HOME ?moving_card)
			; UPDATE HOME_CARD
            (not (HOME ?home_card))
		)
	)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;	WHATEVER FUNCTIONS
;
;		WORKS REGARDLESS
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

	(:action move_card_from_freecell_to_home
	 	:parameters (
			?moving_card ?home_card -card
			?moving_card_num ?home_card_num ?max_cell_available ?current_cell -num
			?suit -suit
		)
	 	:precondition (and
			; CHECK IF MOVING_CARD CAN BE MOVED AND IS IN FREECELL
			(INCELL ?moving_card)
			(CELLSPACE ?max_cell_available)
			(SUCCESSOR ?current_cell ?max_cell_available)
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
	
)

