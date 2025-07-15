(** Training loop for RL agent using DQN or A2C. *)

open Core
open Owl

let train
    ~env
    ~agent
    ~buffer
    ~episodes
    ~batch_size
    ~gamma
  =
  for ep = 1 to episodes do
    (* 1) interact: collect transitions via env.step *)
    (* 2) store in buffer *)
    (* 3) sample from buffer and update agent *)
    ()
  done;
  (* Return training logs or final agent *)
  agent
