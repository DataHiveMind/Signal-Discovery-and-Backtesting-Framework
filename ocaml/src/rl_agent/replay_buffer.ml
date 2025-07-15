(** Simple ring-buffer for experience replay. *)

type transition = {
  state  : Mat.mat;
  action : int;
  reward : float;
  next_state : Mat.mat;
  done   : bool;
}

type t = {
  capacity : int;
  buffer   : transition Array.t;
  mutable idx : int;
  mutable size : int;
}

let create capacity =
  {
    capacity;
    buffer = Array.create ~len:capacity {
      state = Mat.empty 0 0;
      action = 0;
      reward = 0.;
      next_state = Mat.empty 0 0;
      done = false
    };
    idx = 0;
    size = 0;
  }

let add buf tr =
  buf.buffer.(buf.idx) <- tr;
  buf.idx <- (buf.idx + 1) mod buf.capacity;
  buf.size <- Int.min buf.capacity (buf.size + 1)

let sample buf batch_size =
  let open Core in
  let indices = List.init batch_size ~f:(fun _ ->
    Random.int buf.size)
  in
  List.map indices ~f:(fun i -> buf.buffer.(i))
