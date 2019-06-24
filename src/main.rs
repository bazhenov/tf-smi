use clap::{Arg, App};
use tensorflow::{Graph, Session, SessionOptions, Operation};
use std::error::Error;
use std::str::Utf8Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
	let app = App::new("tf-smi")
		.version("1.0")
		.author("Denis Bazhenov <dotsid@gmail.com>")
		.about("Tensorflow Saved Saved Model Inspector")
		.arg(Arg::with_name("model")
				 .short("m")
				 .long("model")
				 .takes_value(true)
				 .required(true)
				 .value_name("DIRECTORY"))
		.get_matches();

	let model_path = app.value_of("model").unwrap();

	inspect(Path::new(model_path))
}

fn inspect(path: &Path) -> Result<(), Box<dyn Error>> {
	let g = Graph::new();

	let mut graph = Graph::new();
	let tags: Vec<&str> = vec!["serve"];
	let opts = SessionOptions::new();
	let session = Session::from_saved_model(&opts, tags, &mut graph, path)?;

	if let Ok(devices) = session.device_list() {
		for ref d in devices {
			println!("{}", d.device_type);
			println!("{}", d.name);
		}
	}

	for o in graph.operation_iter() {
		report_op(&o, &g);
	}

	//let op = graph.operation_by_name_required("embedding_input")?;
	//report_op(&op);
	Ok(())
}

fn report_op(op: &Operation, g: &Graph) -> Result<(), Utf8Error> {
	println!("{}", op.name()?);
	// println!("   input type: {}", op.input_type(0));
	// println!("   output type: {}", op.output_type(0));
	// println!("   op_type: {}", op.op_type()?);
	// println!("   num_control_inputs: {}", op.num_control_inputs());
	// println!("   num_control_outputs: {}", op.num_control_outputs());
	// println!("   num_inputs: {}", op.num_inputs());
	// println!("   num_outputs: {}", op.num_outputs());
	// if let Ok(shape) = op.get_attr_shape("shape") {
	// 	println!("   shape: {}", shape);
	// }
	// if let Ok(dtype) = op.get_attr_type("dtype") {
	// 	println!("   type: {}", dtype);
	// }

	// if let Ok(v) = op.get_attr_tensor::<f32>("value") {
	// 	println!("   value: {}", v);
	// 	println!("   dims: {}", format_dimensions(&v.dims()));
	// }
	
	Ok(())
}

fn format_dimensions(dims: &[u64]) -> String {
	dims.iter()
		.map(|i| i.to_string())
		.collect::<Vec<String>>()
		.join("x")
}

mod tests {

	use super::*;

	#[test]
	fn test_format_dimensions() {
		assert_eq!(format_dimensions(&[1, 2, 3]), "1x2x3");
	}
}