(** Utility functions for data conversion and metrics. *)

open Core
open Owl

(** Flatten a 2D Owl matrix to a float array array *)
let mat_to_array (m : Mat.mat) : float array array =
  List.init (Mat.row_num m) ~f:(fun i ->
    Array.init (Mat.col_num m) ~f:(fun j -> Mat.get m i j)
  )
  |> Array.of_list

(** Compute Sharpe ratio *)
let sharpe ~returns ~risk_free_rate : float =
  let open Float in
  let mean_ret = Statistics.mean returns in
  let sd_ret   = Statistics.std returns in
  (mean_ret - risk_free_rate) / sd_ret
