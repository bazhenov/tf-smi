use clap::{Arg, App};
use tensorflow::{Graph, Session, SessionOptions, Operation, Shape};
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
		.arg_from_usage("[report_shape] -s 'Show the shape of the tensor'")
		.arg_from_usage("[report_type] -t 'Show the type of the tensor'")
		.get_matches();

	let model_path = app.value_of("model").unwrap();

	let opts = Opts {
		report_shape: app.is_present("report_shape"),
		report_type: app.is_present("report_type")
	};

	inspect(Path::new(model_path), &opts)
}

#[derive(Debug)]
struct Opts {
	report_shape: bool,
	report_type: bool
}

fn inspect(path: &Path, opts: &Opts) -> Result<(), Box<dyn Error>> {
	let g = Graph::new();

	let mut graph = Graph::new();
	let tags: Vec<&str> = vec!["serve"];
	let session = Session::from_saved_model(&SessionOptions::new(), tags, &mut graph, path)?;

	if let Ok(devices) = session.device_list() {
		for ref d in devices {
			println!("{}", d.device_type);
			println!("{}", d.name);
		}
	}

	for o in graph.operation_iter() {
		report_op(&o, opts);
	}

	//let op = graph.operation_by_name_required("embedding_input")?;
	//report_op(&op);
	Ok(())
}

fn report_op(op: &Operation, opts: &Opts) -> Result<(), Utf8Error> {
	println!("{}", op.name()?);
	if opts.report_shape {
		if let Ok(shape) = op.get_attr_shape("shape")  {
			println!("   shape: {}", format_shape(&shape));
		}
	}
	if opts.report_type {
		if let Ok(dtype) = op.get_attr_type("dtype") {
			println!("   type: {}", dtype);
		}
	}

	// if let Ok(v) = op.get_attr_tensor::<f32>("value") {
	// 	println!("   value: {}", v);
	// 	println!("   dims: {}", format_dimensions(&v.dims()));
	// }
	
	Ok(())
}

fn format_shape(shape: &Shape) -> String {
	if let Some(size) = shape.dims() {
		let mut s = String::new();
		for i in 0..size {
			if i > 0 {
				s.push_str(", ");
			}
			s.push_str(&shape[i].unwrap_or(-1).to_string());
		}
		s
	} else {
		"-1".to_string()
	}
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

	#[test]
	fn test_format_shape() {
		assert_eq!(format_shape(&Shape::from(Some(vec![Some(1), Some(2)]))), "1, 2");
		assert_eq!(format_shape(&Shape::from(Some(vec![None, Some(2)]))), "-1, 2");
		assert_eq!(format_shape(&Shape::from(None)), "-1");
		assert_eq!(format_shape(&Shape::from(Some(vec![]))), "");
	}
}