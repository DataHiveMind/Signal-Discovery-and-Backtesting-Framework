(** A simple event-driven backtester. *)

open Core
open Owl

type bar = {
  timestamp : float;    (* epoch seconds *)
  features  : Mat.mat;  (* matrix of features per time slice *)
}

(** run ~bars ~initial_cash ~slippage returns PnL series *)
let run
    ~(bars : bar array)
    ~(initial_cash : float)
    ~(slippage : float)
  : float array =
  (* Placeholder: implement event loop, position tracking, pnl calc *)
  Array.map bars ~f:(fun _ -> 0.0)
