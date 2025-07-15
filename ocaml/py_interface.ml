(** Ctypes bindings to expose OCaml functions to Python. *)

open Ctypes
open Foreign

let () =
  let lib = Dl.dlopen ~filename:"_quant.so" ~flags:[Dl.RTLD_NOW] in

  let run_fn =
    foreign ~from:lib "run"
      (ptr double @-> int64_t @-> double @-> returning (ptr double))
  in
  (* Additional FFI exports here *)
  ()
