function gaussPdf = gaussmvnpdf(X, mu, sigma)
    Xmu = X-mu;
    gaussPdf = 1/sqrt(det(sigma)*(2*pi)^2) * exp(-0.5*diag(Xmu*inv(sigma)*Xmu'));
end