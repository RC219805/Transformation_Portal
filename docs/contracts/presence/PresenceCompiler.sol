// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract PresenceCompiler {
    enum Level { None, Concept, Practitioner, Studio, Enterprise }

    address public owner;

    struct ManifestProof {
        address submitter;
        bytes32 manifestHash; // sha3_256(manifest JSON), supplied by the off-chain anchor payload
        bytes32 assetsHash;   // sha3_256(hero||web), supplied by the off-chain anchor payload
        uint64  timestamp;
        Level   level;
    }

    mapping(address => Level) public licenses;
    mapping(bytes32 => ManifestProof) public manifests;

    event Licensed(address indexed who, Level level);
    event Compiled(bytes32 indexed manifestId, address indexed who, Level level);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier requiresLicense(Level minLevel) {
        require(uint(licenses[msg.sender]) >= uint(minLevel), "License level too low");
        _;
    }

    function setLicense(address who, Level level) external onlyOwner {
        licenses[who] = level;
        emit Licensed(who, level);
    }

    function compilePortrait(bytes32 manifestHash, bytes32 assetsHash)
        external requiresLicense(Level.Studio) returns (bytes32 manifestId)
    {
        manifestId = keccak256(abi.encodePacked(manifestHash, assetsHash, msg.sender, block.timestamp));
        manifests[manifestId] = ManifestProof({
            submitter: msg.sender,
            manifestHash: manifestHash,
            assetsHash: assetsHash,
            timestamp: uint64(block.timestamp),
            level: licenses[msg.sender]
        });
        emit Compiled(manifestId, msg.sender, licenses[msg.sender]);
        return manifestId;
    }
}
