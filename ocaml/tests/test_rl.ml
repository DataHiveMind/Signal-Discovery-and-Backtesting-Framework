open OUnit2
open Owl
open Rl_agent.Q_network

let test_q_forward _ =
  let input_dim = 4 and hidden = 8 and output = 2 in
  let net = create ~input_dim ~hidden_dim:hidden ~output_dim:output in
  let x = Dense.Ndarray.D.zeros [|1; input_dim|] in
  let y = forward net x in
  let shape = Dense.Ndarray.S.shape y in
  assert_equal shape [|1; output|]

let suite =
  "rl_agent" >::: [
    "q_forward" >:: test_q_forward
  ]

let () = run_test_tt_main suite
