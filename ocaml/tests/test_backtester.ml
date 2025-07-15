open OUnit2
open Backtester

let make_dummy_bars n =
  Array.init n ~f:(fun i ->
    { timestamp = float_of_int i
    ; features  = Owl.Dense.Matrix.D.ones 1 3
    }
  )

let test_run_no_error _ =
  let bars = make_dummy_bars 10 in
  let pl = run ~bars ~initial_cash:1000. ~slippage:0.0 in
  assert_equal (Array.length pl) 10

let suite =
  "backtester" >::: [
    "run_no_error" >:: test_run_no_error
  ]

let () = run_test_tt_main suite
