# ---------------------------------------------------------------------------- #
#                             step-by-step guide                               #
# ---------------------------------------------------------------------------- #

## caricare il pacchetto Audio911
using Audio911

## load an audiofile

# caricare il file audio da cui si vogliono estrarre features.
# il file audio deve essere dei formati .wav, .mp3, .flac o .ogg,
# in caso di file audio stereo, verranno automaticamente convertiti in mono.
# fare sempre molta attenzione alla frequenza di campionamento del file audio:
# se si stanno usando file campionati a 44100, quindi con un range di frequenza 
# da 0 a 22000 Hz, è veramente necessario computare l'estrazione delle features
# per tutta questa banda?
# se poi, successivamente, si imposta un limite di range tipo: `freqrange=[100, 1000]`
# allora è imperativo impostare una corretta frequenza di campionamento che non ecceda
# al range deciso.
# questo porterà ad un notevole aumento delle prestazioni.

# è anche possibile caricare matrici o dataframe di file audio,
# oppure direttamente un file .csv.
# entrambi i casi verranno affrontati nel prossimo tutorial.

# nella cartella `test/test_files` si trovano dei file audio di esempio.
test_file = joinpath(dirname(@__FILE__), "test_files/test.wav")

audio = load(test_file)

# buona pratica sarebbe quella di specificare sempre tutti i parametri:
# nel caso qualcosa andasse storto è più facile risalire al problema.
# altra importantissima buona pratica è quella si specificare sempre il pacchetto da cui 
# proviene la funzione.
# in questo caso è fondamentale, visto che la funzione `load` è definita in tantissimi
# altri pacchetti di julia.
# quindi il metodo consigliato per caricare correttamente un file audio è il seguente:

audio = Audio911.load(test_file; sr=8000, format=Float32, norm=true)

# - sr: se la frequenza di campionamento originale del file è differente, ricampionarlo a 8000 Hz
# - format: il formato può essere `Float32` o `Float64`.
# - norm: si richiede di normalizzare l'audio (aumentare, se possibile, il volume al massimo)

## from time-domain to frequency domain using fast Fourier transform

# la trasformata di Fourier necessita che l'audio sia dapprima diviso in frames

audioframes = Audio911.Frames(audio; winsize=512, winstep=492, type=hanning)

# - winsize: dimensione della finestra in campioni, si ricorda che più grande è la finestra, 
# maggiore sarà la risoluzione in frequenza e minore la risoluzione temporale.
# si consiglia di partire con una finestra di 256 punti, se la frequenza di campionamento
# è inferiore a 8000hz, oppure 512 se la frequenza di campionamento è superiore a 8000hz
# - winstep: avanzamento della finestra (deve sempre essere inferiore alla winsize)
# - type: tipo di finestra adottata, vedi: https://docs.juliadsp.org/stable/windows/

# quindi è possibile calcolare la stft passandogli i frames

stft_spec = Audio911.Stft(audioframes, nfft=1024, spectrum=power)

# - nfft: dimensione finestra fft, teoricamente la dimensione deve essere uguale a quella della winsize,
# ma è possibile sperimentare con finestre fft più grandi.
# - spectrum: tipo di normalizzazione dello spettro: power(default) più vicino all'ascolto umano,
# magnitude: nessuna normalizzazione

# oppure, più conciso:

stft_spec = Audio911.Stft(audio; nfft=1024, winsize=512, winstep=492, type=hanning, spectrum=power)

# siamo nel dominio delle frequenze e possiamo plottarne il risultato.
# Audio911 definisce le ricette per Plots: basta caricare Plots e chiamare `plot`
# (Plots non è una dipendenza del pacchetto, va aggiunto al proprio ambiente):
#
#   using Plots
#   plot(stft_spec; freq_scale=:log10)

# lo stesso spettrogramma si può ottenere con una trasformata wavelet al posto della stft,
# e tutto ciò che segue accetta indifferentemente l'una o l'altra:

cwt_spec = Audio911.Cwt(audioframes; voices=12, freqrange=(50, 4000))

# lo spettrogramma generato dalla stft è ancora troppo dettagliato per poter permettere un analisi efficace.
# soprattutto è lineare, mentre l'esperienza ha insegnato che uno spettrogramma logaritmico,
# essendo più coerente rispetto al tipo di ascolto dell'essere umano, riesce ad essere molto più preciso.

# per fare questo avremo bisogno innanzitutto di un banco filtro, ovvero una struttura dati che 
# suddivide le frequenze in modo logaritmico:

fbank = auditory_fbank(8000; nfft=1024, nbands=26, norm=bandwidth, domain=:linear, freqrange=(100,1000))

# ora possiamo costruirci un nuovo spettrogramma più preciso e leggero: scegliamo uno spettrogramma di tipo Mel
mel_spec = MelSpec(stft_spec, fbank; win_norm=true)

# o più conciso, senza bisogno di calcolare il filterbank in anticipo:
mel_spec = MelSpec(stft_spec; win_norm=true, freqrange=(100,1000), nbands=26, norm=bandwidth, domain=:linear, scale=htk)

# e infine i coefficienti cepstrali, con le loro derivate temporali:
mfcc  = Mfcc(mel_spec; ncoeffs=13, rect=mlog)
delta = Delta(mfcc)

# gli stessi coefficienti come li calcolerebbero HTK, Kaldi o librosa:
mfcc_k = mfcc_kaldi(audio)
mfcc_l = mfcc_librosa(audio)

# ogni stadio conosce la propria griglia temporale:
get_times(mfcc)
