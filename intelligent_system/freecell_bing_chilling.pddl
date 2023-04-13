(define (domain freecell)
  	(:requirements :strips :typing)  
  	(:types card num suit)
  	(:predicates
		;; CARDS CHARACTERISTICS
	    (VALUE ?c -card ?v -num)
	    (SUCCESSOR ?n1 -num ?n0 -num)
	    (SUIT ?c -card ?s -suit)
	    (CANSTACK ?c1 -card ?c2 -card)
		;; INIT CARDS & FINAL OBJECTIVE CARDS
	    (HOME ?c -card)
		;; CELL CHARACTERISTICS
	    (CELLSPACE ?n -num) ;; # of cells
	    (COLSPACE ?n -num) ;; cols max height
		;; CARDS STATUS
  		(ON ?c1 -card ?c2 -card) 
	    (CLEAR ?c -card) ;; can stack
	    (BOTTOMCOL ?c -card)
		;; CARD STATUS INTERNAL USE
	    (INCELL ?c -card)
	)

  	(:action move_card
		:parameters (
			?card ?old_card ?new_card -card
		)
		:precondition (and 
			(CLEAR ?card)
			(CANSTACK ?card ?new_card)
			(CLEAR ?new_card)
			(ON ?card ?old_card)
		)
		:effect (and
		    (ON ?card ?new_card)
		    (CLEAR ?old_card)
		    (not (ON ?card ?old_card))
		    (not (CLEAR ?new_card))
		)
	)

  	(:action move_card_stack
	   	:parameters (
			?card ?new_card -card
			?cols ?ncols -num
		)
	   	:precondition (and
			(CLEAR ?card) 
			(BOTTOMCOL ?card)
			(CANSTACK ?card ?new_card)
			(COLSPACE ?cols)
			(CLEAR ?new_card)
			(SUCCESSOR ?ncols ?cols)
		)
	   	:effect (and
	   		(ON ?card ?new_card)
			(COLSPACE ?ncols)
			(not (CLEAR ?new_card))
			(not (BOTTOMCOL ?card))
			(not (COLSPACE ?cols))
		)
	)

	(:action send_to_new_col
	 	:parameters (
			?card ?old_card -card
			?cols ?ncols -num
		)
	 	:precondition (and
			(CLEAR ?card)
			(COLSPACE ?cols)
			(SUCCESSOR ?cols ?ncols)
			(ON ?card ?old_card)
		)
	 	:effect (and
		  	(BOTTOMCOL ?card) 
		  	(CLEAR ?old_card)
		  	(COLSPACE ?ncols)
		  	(not (ON ?card ?old_card))
		  	(not (COLSPACE ?cols))
		)
	)

	(:action col_from_free_cell
	 	:parameters (
			?card ?new_card -card
			?cells ?ncells -num
		)
	 	:precondition (and 
			(INCELL ?card)
	        (CANSTACK ?card ?new_card)
		    (CELLSPACE ?cells)
		    (CLEAR ?new_card)
		    (SUCCESSOR ?ncells ?cells)
		)
	 	:effect (and (CELLSPACE ?ncells)
		    (CLEAR ?card)
		    (ON ?card ?new_card)
		    (not (INCELL ?card))
		    (not (CELLSPACE ?cells))
		    (not (CLEAR ?new_card))
		)
	)

  	(:action send_to_free_cell 
	   	:parameters (
			?card ?old_card -card
			?cells ?ncells -num
		)
	   	:precondition (and
			(CLEAR ?card) 
			(ON ?card ?old_card)
			(CELLSPACE ?cells)
			(SUCCESSOR ?cells ?ncells)
		)
	   	:effect (and
		    (CELLSPACE ?ncells)
		    (INCELL ?card) 
		    (CLEAR ?old_card)
		    (not (ON ?card ?old_card))
		    (not (CLEAR ?card))
		    (not (CELLSPACE ?cells))
		)
	)

  	(:action send_to_free_stack 
	   	:parameters (
			?card -card
			?cells ?ncells ?cols ?ncols -num
		)
	   	:precondition (and
			(CLEAR ?card)
	        (COLSPACE ?cols)
			(BOTTOMCOL ?card)
			(SUCCESSOR ?ncols ?cols)
	        (CELLSPACE ?cells)
			(SUCCESSOR ?cells ?ncells)
		)
	   	:effect (and
	        (COLSPACE ?ncols)
		    (CELLSPACE ?ncells)
		    (INCELL ?card)
		    (not (BOTTOMCOL ?card))
		    (not (CLEAR ?card))
		    (not (COLSPACE ?cols))
		    (not (CELLSPACE ?cells))
		)
	)

	(:action free_cell_to_home
	 	:parameters (
			?card ?home_card -card
			?suit -suit
			?vcard ?vhome_card ?cells ?ncells -num
		)
	 	:precondition (and 
			(INCELL ?card)
			(HOME ?home_card) 
			(SUIT ?card ?suit)
			(VALUE ?card ?vcard)
			(VALUE ?home_card ?vhome_card)
			(SUCCESSOR ?vcard ?vhome_card)
			(SUIT ?home_card ?suit)
			(CELLSPACE ?cells)
			(SUCCESSOR ?ncells ?cells)
		)
	 	:effect (and
			(HOME ?card)
			(CELLSPACE ?ncells)
			(not (INCELL ?card))
			(not (CELLSPACE ?cells))
			(not (HOME ?home_card))
		)
	)

	(:action send_card_to_home
		:parameters (
			?card ?old_card ?home_card -card
			?suit -suit
			?vcard ?vhome_card -num
		)
		:precondition (and
			;; CHECK CARD CONDITIONS
			(CLEAR ?card) 
			(ON ?card ?old_card)
			(VALUE ?card ?vcard)
			(SUIT ?card ?suit)
			;; CHECK HOME CONDITIONS
			(HOME ?home_card)
			(SUIT ?home_card ?suit)
			(VALUE ?home_card ?vhome_card)
			(SUCCESSOR ?vcard ?vhome_card)
		)
	 	:effect (and
			;; UPDATE CARDS STATS
		    (not (CLEAR ?card))
		    (CLEAR ?old_card)
		    (not (ON ?card ?old_card))
			;; UPDATE HOME STATS
			(HOME ?card)
            (not (HOME ?home_card))
		)
	)
)

