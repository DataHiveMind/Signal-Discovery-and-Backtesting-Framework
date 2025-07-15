(** Signal generation from feature vectors. *)

open Core
open Owl

type action =
  | Buy of float
  | Sell of float
  | Hold

(** decide_action ~features returns an order *)
let decide_action ~(features : Vec.t) : action =
  (* Placeholder: rule-based or ML-driven signal *)
  Hold
