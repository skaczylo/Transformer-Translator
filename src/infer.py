import os
import torch
from pathlib import Path

# Importamos tus módulos
from config import VOCAB_PATH, BEST_MODEL_PATH
from transformer import TransformerConfig, Transformer
from data import TranslatorTokenizer

def load_model_and_tokenizer(checkpoint_path: Path, vocab_file: Path, device: str):
    """Carga el tokenizador y el modelo con los pesos por defecto."""

    if not vocab_file.exists():
        raise FileNotFoundError(f"No se encontró el vocabulario en: {vocab_file}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontraron los pesos del modelo en: {checkpoint_path}")

    # 1. Cargar Tokenizador
    tokenizer = TranslatorTokenizer(path=str(vocab_file), context_length=128)

    # 2. Configurar Arquitectura
    cfg = TransformerConfig(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_id
    )
    model = Transformer(cfg)

    # 3. Cargar Pesos
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, tokenizer

def display_banner():
    """Imprime un banner con las letras TFG en la terminal."""
    # Códigos de color ANSI
    NARANJA = "\033[38;5;208m"
    GRIS = "\033[90m"
    AZUL = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Arte ASCII generado para "TFG"
    logo = f"""{NARANJA}{BOLD}
 ████████╗███████╗ ██████╗ 
 ╚══██╔══╝██╔════╝██╔════╝ 
    ██║   █████╗  ██║  ███╗
    ██║   ██╔══╝  ██║   ██║
    ██║   ██║     ╚██████╔╝
    ╚═╝   ╚═╝      ╚═════╝ 
{RESET}"""

    # Caja de texto superior adaptada al español
    caja = f"""{GRIS}
 ╭────────────────────────────────────────────────────╮
 │ ✽ ¡Bienvenido a la demo del {RESET}{BOLD}Traductor Transformer{RESET}{GRIS}! │
 ╰────────────────────────────────────────────────────╯{RESET}
"""
    
    print(caja)
    print(logo)
    # También traducimos el mensaje de carga
    print(f" 🎉 {GRIS}Modelo cargado con éxito. ¡Listo para traducir!{RESET}")
    print(f"    {GRIS}Escribe una frase en inglés y pulsa Enter.{RESET}")
    print(f"    {GRIS}Para salir, escribe 'exit', 'quit' o 'salir'.{RESET}\n")
    


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = BEST_MODEL_PATH 
    vocab_file = VOCAB_PATH / "vocab_16k.json"

    # Colores para la consola
    AZUL_CLARO = "\033[96m"
    VERDE = "\033[92m"
    ROJO = "\033[91m"
    RESET = "\033[0m"

    try:
        model, tokenizer = load_model_and_tokenizer(checkpoint_path, vocab_file, device)
        display_banner()

        while True:
            text_to_translate = input(f" {AZUL_CLARO}❯{RESET} Inglés: ")
            if text_to_translate.lower() in ['exit', 'quit', 'salir']:
                print(f"\n👋 ¡Hasta pronto!{RESET}")
                break
            if not text_to_translate.strip():
                continue

            input_ids = tokenizer.encode(text_to_translate, pad=True)
            x = torch.tensor([input_ids], dtype=torch.long).to(device)

            y_pred = model.predict(
                x=x, 
                bos_id=tokenizer.start_id, 
                end_id=tokenizer.end_id, 
                device=device
            )

            translation = tokenizer.decode(y_pred[0].tolist(), skip_special_tokens=True)
            print(f"   {VERDE}Traducción:{RESET} {translation}\n")

    except Exception as e:
        print(f"\n {ROJO}✖ Error:{RESET} {e}\n")

if __name__ == "__main__":
    main()