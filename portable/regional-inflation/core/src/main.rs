use std::io::{self, Read};
fn main() {
    let mut input = String::new();
    if std::env::args().any(|a| a == "--json") {
        if let Some(s) = std::env::args().skip_while(|a| a != "--json").nth(1) {
            input = s
        } else {
            eprintln!("--json requires one JSON argument");
            std::process::exit(2)
        }
    } else if io::stdin().read_to_string(&mut input).is_err() {
        eprintln!("failed to read stdin");
        std::process::exit(2)
    }
    println!("{}", regional_inflation_core::process(&input));
}
