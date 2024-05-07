use darling::FromField;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Data, DataStruct, DeriveInput, Field, Fields, Type};

#[derive(Debug, FromField)]
#[darling(attributes(dev_to_from), default)]
struct AttributeArgs {
    to: bool,
    from: bool,
}

impl Default for AttributeArgs {
    fn default() -> Self {
        Self {
            to: true,
            from: true,
        }
    }
}

fn get_type_string(f: &Field) -> Option<String> {
    match f.ty.clone() {
        Type::Path(type_path) => Some(type_path.clone().into_token_stream().to_string()),
        _ => None,
    }
}

#[proc_macro_derive(DeviceToFrom, attributes(dev_to_from))]
pub fn dev_to_from(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let fields = if let Data::Struct(DataStruct { fields, .. }) = input.data {
        if let Fields::Named(fields_named) = fields {
            fields_named.named
        } else {
            unimplemented!("DeviceToFrom only supports named fields.");
        }
    } else {
        unimplemented!("DeviceToFrom not implemented for enums or unions.");
    };
    let fields = fields.iter().filter(|f| {
        if let Some(s) = get_type_string(f) {
            s.contains("PairedBuffers")
        } else {
            false
        }
    });
    let mut devices_to: Vec<_> = vec![];
    let mut devices_from: Vec<_> = vec![];
    for f in fields {
        let ident = f.ident.clone();
        let attr = AttributeArgs::from_field(f).expect("Could not get attributes from field");
        if !attr.to && !attr.from {
            unimplemented!(
                "PairedBuffers must be at least 'to' or 'from', otherwise they are redundant."
            );
        }
        devices_to.push(match attr.to {
            true => Some(quote!(
                self.#ident.to_device()?;
            )),
            false => None,
        });
        devices_from.push(match attr.from {
            true => Some(quote!(
                self.#ident.from_device()?;
            )),
            false => None,
        });
    }
    quote!(
        impl ::simple_ocl::DeviceToFrom for #name {
            fn send_pairedbuffs(&self) -> ocl::Result<()> {
                #(#devices_to)*
                Ok(())
            }
            fn retrieve_pairedbuffs(&mut self) -> ocl::Result<()> {
                #(#devices_from)*
                Ok(())
            }
        }
    )
    .into()
}
