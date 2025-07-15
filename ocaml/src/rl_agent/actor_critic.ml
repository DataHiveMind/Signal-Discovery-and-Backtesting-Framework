(** Actor-Critic network with separate policy and value heads. *)

open Owl
open Owl.Dense.Ndarray.D

type t = {
  policy_graph : Neural.S.Graph.graph;
  value_graph  : Neural.S.Graph.graph;
}

let create ~input_dim ~hidden_dim ~action_dim =
  let pg = Neural.G.create Graph_network in
  let _ = Neural.G.input pg [| input_dim |] in
  let _ = Neural.G.dense pg hidden_dim in
  let _ = Neural.G.dense pg action_dim in
  Neural.G.build pg;
  let vg = Neural.G.create Graph_network in
  let _ = Neural.G.input vg [| input_dim |] in
  let _ = Neural.G.dense vg hidden_dim in
  let _ = Neural.G.dense vg 1 in
  Neural.G.build vg;
  { policy_graph = pg; value_graph = vg }

let forward_policy t x =
  Neural.S.Graph.run t.policy_graph x

let forward_value t x =
  Neural.S.Graph.run t.value_graph x
