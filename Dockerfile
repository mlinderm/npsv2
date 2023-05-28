FROM tensorflow/tensorflow:2.8.2-gpu

RUN apt-get -qq update && apt-get install --no-install-recommends -yq \
  art-nextgen-simulation-tools \
  bc \
  bcftools \
  bedtools \
  build-essential \
  bwa \
  cmake \
  curl \
  gawk \
  git \
  jellyfish \
  python3-dna-jellyfish \
  libbz2-dev \
  liblzma-dev \
  protobuf-compiler \
  python3-dev \
  python3-pip \
  python3-pkgconfig \
  samtools \
  tabix \
  unzip \
  zlib1g-dev \
  && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/*


RUN mkdir -p /opt/samblaster \
    && curl -SL https://github.com/GregoryFaust/samblaster/releases/download/v.0.1.26/samblaster-v.0.1.26.tar.gz \
    | tar -xzC /opt/samblaster --strip-components=1 \
    && make -C /opt/samblaster \
    && cp /opt/samblaster/samblaster /usr/local/bin/.

RUN curl -SL https://github.com/biod/sambamba/releases/download/v0.8.2/sambamba-0.8.2-linux-amd64-static.gz \
    | gzip -dc > /usr/local/bin/sambamba \
    && chmod +x /usr/local/bin/sambamba

RUN curl -SL https://github.com/brentp/goleft/releases/download/v0.2.4/goleft_linux64 \
    -o /usr/local/bin/goleft \
    && chmod +x /usr/local/bin/goleft

ADD . /opt/npsv2

# Install dependencies
RUN python3 -m pip install --no-cache-dir setuptools==57.5.0
RUN python3 -m pip install --no-cache-dir -r /opt/npsv2/requirements.txt

# Install npsv2
WORKDIR /opt/npsv2
RUN python3 setup.py install