(** Q-network using Owl’s neural modules. *)

open Owl
open Owl.Dense.Ndarray.D

type t = {
  layers : Neural.S.Graph.graph;
  input_shape : int array;
}

let create ~input_dim ~hidden_dim ~output_dim =
  let g = Neural.G.create Graph_network in
  let _ = Neural.G.input g [| input_dim |] in
  let h = Neural.G.dense g hidden_dim in
  let o = Neural.G.dense g output_dim in
  Neural.G.build g;
  { layers = g; input_shape = [| input_dim |] }

let forward t x =
  Neural.S.Graph.run t.layers x
